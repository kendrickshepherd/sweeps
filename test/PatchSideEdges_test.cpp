// Item 19 (outstanding_issues.txt): api::patchSideEdges and api::patchSideElementDarts, the boundary side selection
// behind the four side bindings, against the parametric position of every returned edge and element.  The meshes are
// rectangular in both orders, since a square mesh cannot tell the source map from the line map.
#include <catch2/catch_test_macros.hpp>
#include <NavierStokesDiscretization.hpp>
#include <HierarchicalTPCombinatorialMap.hpp>
#include <CombinatorialMap1d.hpp>
#include <CombinatorialMapMethods.hpp>
#include <ParametricAtlas.hpp>
#include <ParentPoint.hpp>
#include <KnotVector.hpp>
#include <TPSplineSpace.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace basis;
using namespace topology;

namespace
{
    using Index = std::vector<size_t>;

    struct Mesh
    {
        std::string name;
        size_t n_s;
        size_t n_t;
        std::vector<std::vector<std::pair<size_t, size_t>>> marks;
    };

    // Unit cells.  The marks refine the elements at two opposite corners, then a child of one of them at its corner,
    // so every side has leaves on more than one level.
    const std::vector<Mesh> meshes = { { "2 x 3", 2, 3, { { { 1, 2 }, { 0, 0 } }, { { 3, 5 } } } },
                                       { "3 x 2", 3, 2, { { { 2, 1 }, { 0, 0 } }, { { 5, 3 } } } },
                                       { "3 x 3 (square control)", 3, 3, { { { 2, 2 }, { 0, 0 } }, { { 5, 5 } } } } };

    constexpr size_t degree = 2;

    struct Side
    {
        api::PatchSide side;
        std::string name;
        size_t axis; // The coordinate fixed on the side: 0 for s, 1 for t.
        bool upper;
    };

    const std::vector<Side> sides = { { api::PatchSide::S0, "S0", 0, false },
                                      { api::PatchSide::S1, "S1", 0, true },
                                      { api::PatchSide::T0, "T0", 1, false },
                                      { api::PatchSide::T1, "T1", 1, true } };

    double sideLine( const Mesh& mesh, const Side& side )
    {
        return side.upper ? double( side.axis == 0 ? mesh.n_s : mesh.n_t ) : 0.0;
    }

    // An element's parametric box: low corner, then high corner.
    using Box = std::array<double, 4>;
    using Interval = std::pair<double, double>;

    Interval along( const Box& box, const Side& side )
    {
        return side.axis == 0 ? Interval{ box.at( 1 ), box.at( 3 ) } : Interval{ box.at( 0 ), box.at( 2 ) };
    }

    std::string str( const double x )
    {
        std::ostringstream out;
        out << x;
        return out.str();
    }

    std::string describe( const Interval& in ) { return "[" + str( in.first ) + ", " + str( in.second ) + "]"; }

    std::string describe( const Box& box )
    {
        return describe( Interval{ box.at( 0 ), box.at( 2 ) } ) + " x " + describe( Interval{ box.at( 1 ), box.at( 3 ) } );
    }

    template <typename T>
    std::string describe( const std::vector<T>& xs )
    {
        std::string out;
        for( const T& x : xs ) out += ( out.empty() ? "" : " " ) + describe( x );
        return "{" + out + "}";
    }

    std::string join( const std::vector<std::string>& problems )
    {
        std::string out;
        for( const std::string& p : problems ) out += p + "\n";
        return out;
    }

    // Simple interior knots on unit breaks, end knots of multiplicity degree + 1.
    KnotVector openKnots( const size_t n_cells )
    {
        std::vector<double> knots( degree, 0.0 );
        for( size_t i = 0; i <= n_cells; i++ ) knots.push_back( double( i ) );
        knots.insert( knots.end(), degree, double( n_cells ) );
        return KnotVector( knots, 1e-10 );
    }

    Eigen::MatrixXd identityGeometry( const KnotVector& kv_s, const KnotVector& kv_t )
    {
        return grevillePoints( buildBSpline( { kv_s, kv_t }, { degree, degree } ) ).transpose();
    }

    // (level, i, j) of every leaf: all level 0 cells, each mark replaced by its children 2 i + {0, 1}.
    std::set<Index> leavesFromMarks( const Mesh& mesh, const bool refine )
    {
        std::set<Index> leaves;
        for( size_t j = 0; j < mesh.n_t; j++ )
            for( size_t i = 0; i < mesh.n_s; i++ ) leaves.insert( { 0, i, j } );
        if( not refine ) return leaves;
        for( size_t level = 0; level < mesh.marks.size(); level++ )
            for( const auto& [i, j] : mesh.marks.at( level ) )
            {
                if( leaves.erase( { level, i, j } ) != 1 ) throw std::runtime_error( "a mark is not a leaf" );
                for( size_t b = 0; b < 2; b++ )
                    for( size_t a = 0; a < 2; a++ ) leaves.insert( { level + 1, 2 * i + a, 2 * j + b } );
            }
        return leaves;
    }

    // The boxes of the leaves with an edge on the side, in order along it.
    std::vector<Box> expectedBoxes( const Mesh& mesh, const Side& side, const bool refine )
    {
        std::vector<Box> out;
        for( const Index& leaf : leavesFromMarks( mesh, refine ) )
        {
            const double h = 1.0 / double( size_t{ 1 } << leaf.at( 0 ) );
            const Box box = { leaf.at( 1 ) * h, leaf.at( 2 ) * h, ( leaf.at( 1 ) + 1 ) * h, ( leaf.at( 2 ) + 1 ) * h };
            if( box.at( side.axis + ( side.upper ? 2 : 0 ) ) == sideLine( mesh, side ) ) out.push_back( box );
        }
        std::sort( out.begin(), out.end(), [&]( const Box& a, const Box& b ) {
            return along( a, side ) < along( b, side );
        } );
        return out;
    }

    Box elementBox( const param::ParametricAtlas& atlas, const Dart& d )
    {
        const Vector6dMax starts = atlas.parametricStarts( Face( d ) );
        const Vector6dMax lengths = atlas.parametricLengths( Face( d ) );
        return { starts( 0 ), starts( 1 ), starts( 0 ) + lengths( 0 ), starts( 1 ) + lengths( 1 ) };
    }

    // The parametric point of the start vertex of d, in the chart of d's element.
    std::array<double, 2> startPoint( const param::ParametricAtlas& atlas, const Dart& d )
    {
        const param::ParentPoint ppt = atlas.parentPoint( Vertex( d ) );
        const Box box = elementBox( atlas, d );
        return { box.at( 0 ) + ppt.mPoint( 0 ) * ( box.at( 2 ) - box.at( 0 ) ),
                 box.at( 1 ) + ppt.mPoint( 1 ) * ( box.at( 3 ) - box.at( 1 ) ) };
    }

    // Both entry points for one side of one map.  Each returned edge must be a dart of the map on the boundary with
    // both end points on the side; the edges must cover the expected leaves' intervals along the side, and the element
    // darts must be the lowest darts of the expected leaves and of the edges' elements.  Ordered: in order along the
    // side, else as multisets.  All coordinates are dyadic, so they are compared exactly.
    template <typename Map>
    std::vector<std::string> sideProblems( const std::string& where,
                                           const Map& cmap,
                                           const param::ParametricAtlas& atlas,
                                           const Mesh& mesh,
                                           const Side& side,
                                           const std::vector<Box>& expected,
                                           const bool ordered )
    {
        std::vector<std::string> problems;
        std::set<Dart::IndexType> darts;
        iterateDartsWhile( cmap, [&]( const Dart& d ) {
            darts.insert( d.id() );
            return true;
        } );
        const double line = sideLine( mesh, side );
        const size_t across = 1 - side.axis;
        const auto sorted = [&]( auto xs ) {
            if( not ordered ) std::sort( xs.begin(), xs.end() );
            return xs;
        };

        std::optional<std::vector<Edge>> edges;
        try
        {
            edges = api::patchSideEdges( cmap, side.side );
            std::vector<Interval> covered;
            for( const Edge& e : edges.value() )
            {
                const std::string edge = where + " edge " + std::to_string( e.dart().id() );
                if( darts.count( e.dart().id() ) == 0 )
                {
                    problems.push_back( edge + " is not a dart of the map" );
                    continue;
                }
                if( phi( cmap, 2, e.dart() ).has_value() ) problems.push_back( edge + " is not on the boundary" );
                const auto p = startPoint( atlas, e.dart() );
                const auto q = startPoint( atlas, phi( cmap, 1, e.dart() ).value() );
                if( p.at( side.axis ) != line or q.at( side.axis ) != line )
                    problems.push_back( edge + " runs from (" + str( p.at( 0 ) ) + ", " + str( p.at( 1 ) ) + ") to (" +
                                        str( q.at( 0 ) ) + ", " + str( q.at( 1 ) ) + "), off the side" );
                covered.push_back( { std::min( p.at( across ), q.at( across ) ), std::max( p.at( across ), q.at( across ) ) } );
            }
            std::vector<Interval> wanted;
            for( const Box& box : expected ) wanted.push_back( along( box, side ) );
            if( sorted( covered ) != sorted( wanted ) )
                problems.push_back( where + " edges cover " + describe( covered ) + ", not " + describe( wanted ) );
        }
        catch( const std::exception& e )
        {
            problems.push_back( where + " patchSideEdges threw: " + e.what() );
            edges.reset();
        }

        try
        {
            const std::vector<Dart::IndexType> elements = api::patchSideElementDarts( cmap, side.side );
            std::vector<Box> boxes;
            for( const Dart::IndexType id : elements )
            {
                const std::string element = where + " element dart " + std::to_string( id );
                if( darts.count( id ) == 0 )
                {
                    problems.push_back( element + " is not a dart of the map" );
                    continue;
                }
                const Dart::IndexType lowest = lowestDartId( cmap, Face( Dart( id ) ) );
                if( lowest != id ) problems.push_back( element + " is not its element's lowest dart " + std::to_string( lowest ) );
                boxes.push_back( elementBox( atlas, Dart( id ) ) );
            }
            if( sorted( boxes ) != sorted( expected ) )
                problems.push_back( where + " element darts are on " + describe( boxes ) + ", not " + describe( expected ) );

            if( edges and edges->size() != elements.size() )
                problems.push_back( where + " " + std::to_string( edges->size() ) + " edges but " +
                                    std::to_string( elements.size() ) + " element darts" );
            else if( edges )
                for( size_t k = 0; k < elements.size(); k++ )
                {
                    const Dart& d = edges->at( k ).dart();
                    if( darts.count( d.id() ) == 1 and lowestDartId( cmap, Face( d ) ) != elements.at( k ) )
                        problems.push_back( where + " element dart " + std::to_string( elements.at( k ) ) +
                                            " is not the element of edge " + std::to_string( d.id() ) );
                }
        }
        catch( const std::exception& e )
        {
            problems.push_back( where + " patchSideElementDarts threw: " + e.what() );
        }
        return problems;
    }

    template <typename F>
    std::string outcome( const F& call )
    {
        try
        {
            call();
        }
        catch( const std::invalid_argument& e )
        {
            return std::string( "invalid_argument: " ) + e.what();
        }
        catch( const std::exception& e )
        {
            return std::string( "other exception: " ) + e.what();
        }
        return "returned";
    }
} // namespace

TEST_CASE( "Patch side edges and element darts on rectangular meshes and a square control" )
{
    for( const Mesh& mesh : meshes )
    {
        const KnotVector kv_s = openKnots( mesh.n_s );
        const KnotVector kv_t = openKnots( mesh.n_t );
        const Eigen::MatrixXd cpts = identityGeometry( kv_s, kv_t );

        const api::NavierStokesTPDiscretization tp( kv_s, kv_t, degree, degree, cpts );
        const api::NavierStokesHierarchicalDiscretization h( kv_s, kv_t, degree, degree, cpts, mesh.marks );
        const auto& tp_atlas = tp.H1_ss.basisComplex().parametricAtlas();
        const auto& h_atlas = h.H1_ss.basisComplex().parametricAtlas();
        const TPCombinatorialMap& tp_map = tp_atlas.cmap();
        const HierarchicalTPCombinatorialMap& h_map = h_atlas.cmap();

        // On a rectangular mesh these establish that the source map runs in s and the line map in t.
        for( const TPCombinatorialMap* level0 : { &tp_map, h_map.refinementLevels().front().get() } )
        {
            REQUIRE( cellCount( level0->sourceCMap(), 1 ) == mesh.n_s );
            REQUIRE( cellCount( level0->lineCMap(), 1 ) == mesh.n_t );
        }

        std::vector<std::string> tp_problems;
        std::vector<std::string> h_problems;
        for( const Side& side : sides )
        {
            const std::vector<std::string> tp_side = sideProblems(
                mesh.name + " tensor product " + side.name, tp_map, tp_atlas, mesh, side, expectedBoxes( mesh, side, false ), true );
            const std::vector<std::string> h_side = sideProblems(
                mesh.name + " hierarchical " + side.name, h_map, h_atlas, mesh, side, expectedBoxes( mesh, side, true ), false );
            tp_problems.insert( tp_problems.end(), tp_side.begin(), tp_side.end() );
            h_problems.insert( h_problems.end(), h_side.begin(), h_side.end() );
        }
        {
            INFO( join( tp_problems ) );
            CHECK( tp_problems.empty() );
        }
        {
            INFO( join( h_problems ) );
            CHECK( h_problems.empty() );
        }
    }
}

TEST_CASE( "An unknown patch side throws from every entry point" )
{
    const Mesh& mesh = meshes.front();
    const KnotVector kv_s = openKnots( mesh.n_s );
    const KnotVector kv_t = openKnots( mesh.n_t );
    const Eigen::MatrixXd cpts = identityGeometry( kv_s, kv_t );
    const api::NavierStokesTPDiscretization tp( kv_s, kv_t, degree, degree, cpts );
    const api::NavierStokesHierarchicalDiscretization h( kv_s, kv_t, degree, degree, cpts, mesh.marks );
    const TPCombinatorialMap& tp_map = tp.H1_ss.basisComplex().parametricAtlas().cmap();
    const HierarchicalTPCombinatorialMap& h_map = h.H1_ss.basisComplex().parametricAtlas().cmap();

    const api::PatchSide bad = static_cast<api::PatchSide>( 7 );
    const std::string expected = "invalid_argument: Unknown patch side.";
    CHECK( outcome( [&] { api::patchSideEdges( tp_map, bad ); } ) == expected );
    CHECK( outcome( [&] { api::patchSideElementDarts( tp_map, bad ); } ) == expected );
    CHECK( outcome( [&] { api::patchSideEdges( h_map, bad ); } ) == expected );
    CHECK( outcome( [&] { api::patchSideElementDarts( h_map, bad ); } ) == expected );
}

TEST_CASE( "A 3d map has no patch sides" )
{
    const auto line_1 = std::make_shared<const CombinatorialMap1d>( 1 );
    const auto line_2 = std::make_shared<const CombinatorialMap1d>( 2 );
    const auto volume = std::make_shared<const TPCombinatorialMap>(
        std::make_shared<const TPCombinatorialMap>( line_1, line_1 ), line_2 );
    const HierarchicalTPCombinatorialMap h_volume( { volume }, { { Volume( 0 ), Volume( 24 ) } } );
    REQUIRE( volume->dim() == 3 );
    REQUIRE( h_volume.dim() == 3 );

    const std::string expected = "invalid_argument: Patch sides are defined only for 2d tensor-product maps.";
    for( const Side& side : sides )
    {
        INFO( side.name );
        CHECK( outcome( [&] { api::patchSideEdges( *volume, side.side ); } ) == expected );
        CHECK( outcome( [&] { api::patchSideElementDarts( *volume, side.side ); } ) == expected );
        CHECK( outcome( [&] { api::patchSideEdges( h_volume, side.side ); } ) == expected );
        CHECK( outcome( [&] { api::patchSideElementDarts( h_volume, side.side ); } ) == expected );
    }
}
