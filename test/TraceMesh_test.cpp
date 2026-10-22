#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <HierarchicalMultiPatchSplineSpace.hpp>
#include <MultiPatchSplineFactory.hpp>
#include <TraceMesh.hpp>
#include <VectorConformingHierarchicalMultiPatchSplineSpace.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>

using namespace basis;
using namespace topology;

namespace
{
    using InternalConnectionsMap = MultiPatchCombinatorialMap::InternalConnectionsMap;
    using ConstituentSide = MultiPatchCombinatorialMap::ConstituentSide;
    using TPPermutation = MultiPatchCombinatorialMap::TPPermutation;

    Cell topCellAt( const TPCombinatorialMap& cmap, const std::vector<size_t>& element_indices )
    {
        FullyUnflattenedDart unflat;
        for( const size_t element_index : element_indices )
        {
            unflat.unflat_darts.push_back( Dart( element_index ) );
        }
        unflat.dart_pos =
            SmallVector<TPCombinatorialMap::TPDartPos, 2>(
                element_indices.size() - 1, TPCombinatorialMap::TPDartPos::DartPos0 );
        return Cell( flattenFull( cmap, unflat ), cmap.dim() );
    }

    Cell patchTopCellAt( const MultiPatchCombinatorialMap& cmap,
                         const size_t patch_id,
                         const std::vector<size_t>& element_indices )
    {
        const Cell local = topCellAt( *cmap.constituents().at( patch_id ), element_indices );
        return Cell( cmap.toGlobalDart( patch_id, local.dart() ), cmap.dim() );
    }

    InternalConnectionsMap twoPatchConnection( const ElementSide& first_side,
                                               const ElementSide& second_side,
                                               const TPPermutation permutation )
    {
        const ConstituentSide first{ 0, first_side.sideId() };
        const ConstituentSide second{ 1, second_side.sideId() };
        return { { first, { permutation, second } }, { second, { permutation, first } } };
    }

    std::shared_ptr<const TPSplineSpace> makePatch( const SmallVector<KnotVector, 3>& kvs,
                                                    const SmallVector<size_t, 3>& degrees )
    {
        return std::make_shared<const TPSplineSpace>( buildBSpline( kvs, degrees ) );
    }

    MultiPatchSplineSpace makeTwoPatchSurface()
    {
        const double ptol = 1e-10;
        const KnotVector kv_s( { 0, 0, 0, 1, 1, 1 }, ptol );
        const KnotVector kv_t_c0( { 0, 0, 0, 1, 1, 2, 2, 2 }, ptol );
        const auto patch = makePatch( { kv_s, kv_t_c0 }, { 2, 2 } );
        return buildH1MultiPatchSplineSpace(
            { patch, patch },
            twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ), TPPermutation::Flip1d ) );
    }

    MultiPatchSplineSpace makeTwoPatchVolume()
    {
        const double ptol = 1e-10;
        const KnotVector kv( { 0, 0, 0, 1, 1, 1 }, ptol );
        const auto patch = makePatch( { kv, kv, kv }, { 2, 2, 2 } );
        return buildH1MultiPatchSplineSpace(
            { patch, patch },
            twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ), TPPermutation::ZeroToZero ) );
    }

    HierarchicalMultiPatchSplineSpace makeMixedLevelTwoPatchHierarchy()
    {
        const double ptol = 1e-10;
        const KnotVector coarse_s( { 0, 0, 0, 1, 2, 2, 2 }, ptol );
        const KnotVector fine_s( { 0, 0, 0, 0.5, 1, 1.5, 2, 2, 2 }, ptol );
        const KnotVector t( { 0, 0, 0, 1, 1, 1 }, ptol );

        const auto coarse_patch = makePatch( { coarse_s, t }, { 2, 2 } );
        const auto fine_patch = makePatch( { fine_s, t }, { 2, 2 } );
        const InternalConnectionsMap connections =
            twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ), TPPermutation::Flip1d );

        auto coarse_h1 = std::make_shared<const MultiPatchSplineSpace>(
            buildH1MultiPatchSplineSpace( { coarse_patch, coarse_patch }, connections ) );
        auto fine_h1 = std::make_shared<const MultiPatchSplineSpace>(
            buildH1MultiPatchSplineSpace( { fine_patch, fine_patch }, connections ) );

        const auto& coarse_cmap =
            dynamic_cast<const MultiPatchCombinatorialMap&>(
                coarse_h1->basisComplex().parametricAtlas().cmap() );
        const auto& fine_cmap =
            dynamic_cast<const MultiPatchCombinatorialMap&>(
                fine_h1->basisComplex().parametricAtlas().cmap() );

        return buildHierarchicalSplineSpace(
            { coarse_h1, fine_h1 },
            {
                {
                    patchTopCellAt( coarse_cmap, 0, { 0, 0 } ),
                    patchTopCellAt( coarse_cmap, 1, { 1, 0 } )
                },
                {
                    patchTopCellAt( fine_cmap, 0, { 2, 0 } ),
                    patchTopCellAt( fine_cmap, 0, { 3, 0 } ),
                    patchTopCellAt( fine_cmap, 1, { 0, 0 } ),
                    patchTopCellAt( fine_cmap, 1, { 1, 0 } )
                }
            } );
    }

    size_t sharedTraceIdCount( const TraceInterfaceElement& iface )
    {
        REQUIRE( iface.second.has_value() );
        std::set<FunctionId> first( iface.first.connectivity.begin(), iface.first.connectivity.end() );
        std::set<FunctionId> second( iface.second->connectivity.begin(), iface.second->connectivity.end() );
        std::vector<FunctionId> shared;
        std::set_intersection( first.begin(), first.end(), second.begin(), second.end(), std::back_inserter( shared ) );
        return shared.size();
    }

    size_t aggregateSharedTraceIdCount( const SplineSpace& ss,
                                        const std::vector<TraceMeshInterface>& interfaces )
    {
        std::set<FunctionId> first_ids;
        std::set<FunctionId> second_ids;
        for( const TraceMeshInterface& iface : interfaces )
        {
            const TraceInterfaceElement trace = traceInterfaceElement( ss, iface );
            REQUIRE( trace.second.has_value() );
            first_ids.insert( trace.first.connectivity.begin(), trace.first.connectivity.end() );
            second_ids.insert( trace.second->connectivity.begin(), trace.second->connectivity.end() );
        }

        std::vector<FunctionId> shared;
        std::set_intersection( first_ids.begin(),
                               first_ids.end(),
                               second_ids.begin(),
                               second_ids.end(),
                               std::back_inserter( shared ) );
        return shared.size();
    }
}

TEST_CASE( "Phase 5A trace side coordinate permutations match multipatch conventions" )
{
    // Side pairs chosen so F_b o P o F_a equals P, which is what these values
    // were originally pinned against: 2D sides 0<->3 reverse, and 3D sides
    // 1<->2 leave P untouched.
    const auto t2d = topology::sideCoordinateTransform( 2, 0, 3, TPPermutation::Flip1d );
    CHECK( permuteTraceSideIndex( t2d, { 0 }, { 3 } ) == std::vector<size_t>{ 2 } );
    CHECK_THAT( permuteTraceSidePoint( t2d, Eigen::Vector<double, 1>( 0.25 ) )( 0 ),
                Catch::Matchers::WithinAbs( 0.75, 1e-14 ) );

    const auto t3d = []( const TPPermutation p ) {
        return topology::sideCoordinateTransform( 3, 1, 2, p );
    };

    CHECK( permuteTraceSideIndex( t3d( TPPermutation::ZeroToZero ), { 0, 3 }, { 4, 5 } ) ==
           std::vector<size_t>{ 3, 3 } );
    CHECK( permuteTraceSideIndex( t3d( TPPermutation::ZeroToOne ), { 0, 3 }, { 4, 5 } ) ==
           std::vector<size_t>{ 1, 3 } );
    CHECK( permuteTraceSideIndex( t3d( TPPermutation::ZeroToTwo ), { 0, 3 }, { 4, 5 } ) ==
           std::vector<size_t>{ 0, 1 } );
    CHECK( permuteTraceSideIndex( t3d( TPPermutation::ZeroToThree ), { 0, 3 }, { 4, 5 } ) ==
           std::vector<size_t>{ 3, 0 } );

    const Eigen::Vector2d point( 0.2, 0.7 );
    CHECK( ( permuteTraceSidePoint( t3d( TPPermutation::ZeroToZero ), point ) -
             Eigen::Vector2d( 0.8, 0.7 ) )
               .norm() < 1e-14 );
    CHECK( ( permuteTraceSidePoint( t3d( TPPermutation::ZeroToOne ), point ) -
             Eigen::Vector2d( 0.3, 0.8 ) )
               .norm() < 1e-14 );
    CHECK( ( permuteTraceSidePoint( t3d( TPPermutation::ZeroToTwo ), point ) -
             Eigen::Vector2d( 0.2, 0.3 ) )
               .norm() < 1e-14 );
    CHECK( ( permuteTraceSidePoint( t3d( TPPermutation::ZeroToThree ), point ) -
             Eigen::Vector2d( 0.7, 0.2 ) )
               .norm() < 1e-14 );

    // The side pair now matters: on a disagreeing pair the same permutation
    // gives a different map.  These rows fail under the old P-only consumer.
    const auto t2d_agree = topology::sideCoordinateTransform( 2, 0, 1, TPPermutation::Flip1d );
    CHECK( permuteTraceSideIndex( t2d_agree, { 0 }, { 3 } ) == std::vector<size_t>{ 0 } );
    const auto t3d_flip = topology::sideCoordinateTransform( 3, 0, 1, TPPermutation::ZeroToZero );
    CHECK( permuteTraceSideIndex( t3d_flip, { 0, 3 }, { 4, 5 } ) == std::vector<size_t>{ 0, 3 } );
}

TEST_CASE( "Phase 5A trace mesh enumerates simple 3D patch interfaces" )
{
    const MultiPatchSplineSpace h1 = makeTwoPatchVolume();

    const std::vector<TraceMeshInterface> boundaries = boundaryTraceMeshInterfaces( h1 );
    const std::vector<TraceMeshInterface> patch_interfaces = patchTraceMeshInterfaces( h1 );
    const std::vector<TraceMeshInterface> interiors = interiorTraceMeshInterfaces( h1 );
    const std::vector<TraceMeshInterface> all =
        traceMeshInterfaces( h1, true, true, true );

    CHECK( boundaries.size() == 10 );
    REQUIRE( patch_interfaces.size() == 1 );
    CHECK( interiors.empty() );
    CHECK( all.size() == boundaries.size() + patch_interfaces.size() + interiors.size() );

    const TraceMeshInterface& iface = patch_interfaces.front();
    CHECK( iface.type == TraceMeshEntityType::PatchInterface );
    REQUIRE_FALSE( iface.isBoundary() );
    CHECK( iface.first.patch_id == 0 );
    CHECK( iface.second->patch_id == 1 );
    CHECK( iface.first.side == ElementSide( 0, false ) );
    CHECK( iface.second->side == ElementSide( 0, true ) );
    REQUIRE( iface.permutation.has_value() );
    CHECK( iface.permutation.value() == TPPermutation::ZeroToZero );
    CHECK( sharedTraceIdCount( traceInterfaceElement( h1, iface ) ) == 9 );
}

TEST_CASE( "Phase 5A trace mesh enumerates boundary, patch-interface, and interior sides" )
{
    const MultiPatchSplineSpace h1 = makeTwoPatchSurface();

    const std::vector<TraceMeshInterface> boundaries = boundaryTraceMeshInterfaces( h1 );
    const std::vector<TraceMeshInterface> patch_interfaces = patchTraceMeshInterfaces( h1 );
    const std::vector<TraceMeshInterface> interiors = interiorTraceMeshInterfaces( h1 );
    const std::vector<TraceMeshInterface> broken_interiors =
        interiorTraceMeshInterfaces( h1, InteriorTraceMode::Broken );

    CHECK( boundaries.size() == 8 );
    CHECK( patch_interfaces.size() == 2 );
    CHECK( interiors.size() == 2 );
    CHECK( broken_interiors.empty() );

    for( const TraceMeshInterface& iface : boundaries )
    {
        CHECK( iface.type == TraceMeshEntityType::Boundary );
        CHECK( iface.isBoundary() );
        CHECK( not iface.permutation.has_value() );
        CHECK( traceInterfaceElement( h1, iface ).isBoundary() );
    }

    for( const TraceMeshInterface& iface : patch_interfaces )
    {
        CHECK( iface.type == TraceMeshEntityType::PatchInterface );
        REQUIRE_FALSE( iface.isBoundary() );
        REQUIRE( iface.permutation.has_value() );
        CHECK( iface.permutation.value() == TPPermutation::Flip1d );
        CHECK( iface.first.level == 0 );
        CHECK( iface.second->level == 0 );
        CHECK( iface.first.patch_id != iface.second->patch_id );
        CHECK( sharedTraceIdCount( traceInterfaceElement( h1, iface ) ) > 0 );
    }
    CHECK( aggregateSharedTraceIdCount( h1, patch_interfaces ) == 5 );

    for( const TraceMeshInterface& iface : interiors )
    {
        CHECK( iface.type == TraceMeshEntityType::Interior );
        REQUIRE_FALSE( iface.isBoundary() );
        CHECK( iface.first.patch_id == iface.second->patch_id );
        CHECK( sharedTraceIdCount( traceInterfaceElement( h1, iface ) ) == 3 );
    }
}

TEST_CASE( "Phase 5A broken interior filtering detects low-continuity reduced spaces" )
{
    const MultiPatchSplineSpace h1 = makeTwoPatchSurface();
    const VectorConformingMultiPatchSplineSpace hdiv = buildHDivMultiPatchSplineSpace( h1 );
    const MultiPatchSplineSpace l2 = buildL2MultiPatchSplineSpace( hdiv );

    const std::vector<TraceMeshInterface> interiors = interiorTraceMeshInterfaces( l2 );
    const std::vector<TraceMeshInterface> broken_interiors =
        interiorTraceMeshInterfaces( l2, InteriorTraceMode::Broken );

    REQUIRE( interiors.size() == 2 );
    CHECK( broken_interiors.size() == 2 );
    for( const TraceMeshInterface& iface : broken_interiors )
    {
        CHECK( sharedTraceIdCount( traceInterfaceElement( l2, iface ) ) == 0 );
    }
}

TEST_CASE( "Phase 5A trace mesh enumerates mixed-level hierarchical patch interfaces" )
{
    const HierarchicalMultiPatchSplineSpace h1 = makeMixedLevelTwoPatchHierarchy();
    const VectorConformingHierarchicalMultiPatchSplineSpace hdiv =
        buildHDivHierarchicalMultiPatchSplineSpace( h1 );

    const std::vector<TraceMeshInterface> boundaries = boundaryTraceMeshInterfaces( hdiv );
    const std::vector<TraceMeshInterface> patch_interfaces = patchTraceMeshInterfaces( hdiv );
    const std::vector<TraceMeshInterface> interiors = interiorTraceMeshInterfaces( hdiv );
    const std::vector<TraceMeshInterface> all =
        traceMeshInterfaces( hdiv, true, true, true );

    CHECK( boundaries.size() == 14 );
    REQUIRE( patch_interfaces.size() == 1 );
    CHECK( interiors.size() == 2 );
    CHECK( all.size() == boundaries.size() + patch_interfaces.size() + interiors.size() );

    const TraceMeshInterface& iface = patch_interfaces.front();
    CHECK( iface.type == TraceMeshEntityType::PatchInterface );
    REQUIRE_FALSE( iface.isBoundary() );
    CHECK( iface.first.level == 1 );
    CHECK( iface.second->level == 1 );
    CHECK( iface.first.patch_id == 0 );
    CHECK( iface.second->patch_id == 1 );
    REQUIRE( iface.permutation.has_value() );
    CHECK( iface.permutation.value() == TPPermutation::Flip1d );

    const TraceInterfaceElement trace = traceInterfaceElement( hdiv, iface );
    REQUIRE_FALSE( trace.isBoundary() );
    CHECK( sharedTraceIdCount( trace ) > 0 );
}

// Item 1, step 6 (outstanding_issues.txt): single-level patch-interface records
// come per connection in map order, then per first-side element in
// lexicographic side-index order.  Each side embeds its whole element side,
// the second in the interface's axes.
namespace
{
    std::string elementAt( const MultiPatchCombinatorialMap& cmap, const Cell& element )
    {
        const auto [patch, local_d] = cmap.toLocalDart( element.dart() );
        const FullyUnflattenedDart unflat = unflattenFull( *cmap.constituents().at( patch ), local_d );
        std::string out = "p" + std::to_string( patch ) + " (";
        for( size_t axis = 0; axis < unflat.unflat_darts.size(); axis++ )
            out += ( axis == 0 ? "" : "," ) + std::to_string( unflat.unflat_darts.at( axis ).id() );
        return out + ")";
    }

    std::vector<std::string> patchInterfaceOrder( const MultiPatchSplineSpace& ss )
    {
        const auto& cmap =
            dynamic_cast<const MultiPatchCombinatorialMap&>( ss.basisComplex().parametricAtlas().cmap() );
        std::vector<std::string> out;
        for( const TraceMeshInterface& iface : patchTraceMeshInterfaces( ss ) )
        {
            REQUIRE( iface.second.has_value() );
            out.push_back( elementAt( cmap, iface.first.element ) + " / " + elementAt( cmap, iface.second->element ) );
        }
        return out;
    }

    // Source axis per destination, "r" if reversed; then offsets; then scales.
    std::string embeddingText( const TraceSideEmbedding& e )
    {
        std::ostringstream out;
        for( size_t d = 0; d < e.axes.source_axis_for_destination.size(); d++ )
            out << ( d == 0 ? "" : " " ) << e.axes.source_axis_for_destination.at( d )
                << ( e.axes.source_axis_reversed.at( d ) ? "r" : "" );
        out << " |";
        for( const double o : e.offset ) out << " " << o;
        out << " |";
        for( const double c : e.scale ) out << " " << c;
        return out.str();
    }

    std::set<std::string> patchInterfaceEmbeddings( const MultiPatchSplineSpace& ss )
    {
        std::set<std::string> out;
        for( const TraceMeshInterface& iface : patchTraceMeshInterfaces( ss ) )
        {
            REQUIRE( iface.second.has_value() );
            out.insert( embeddingText( iface.first.embedding ) + " / " + embeddingText( iface.second->embedding ) );
        }
        return out;
    }
}

TEST_CASE( "Single-level patch-interface enumeration order is pinned" )
{
    const double ptol = 1e-10;
    const KnotVector one( { 0, 0, 0, 1, 1, 1 }, ptol );
    const KnotVector two( { 0, 0, 0, 1, 2, 2, 2 }, ptol );
    const KnotVector three( { 0, 0, 0, 1, 2, 3, 3, 3 }, ptol );

    SECTION( "3D axis swap: second index (2 - u, t)" )
    {
        const MultiPatchSplineSpace h1 = buildH1MultiPatchSplineSpace(
            { makePatch( { one, two, three }, { 2, 2, 2 } ), makePatch( { one, three, two }, { 2, 2, 2 } ) },
            twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ), TPPermutation::ZeroToOne ) );
        CHECK( patchInterfaceOrder( h1 ) == std::vector<std::string>{ "p0 (0,0,0) / p1 (0,2,0)",
                                                                      "p0 (0,0,1) / p1 (0,1,0)",
                                                                      "p0 (0,0,2) / p1 (0,0,0)",
                                                                      "p0 (0,1,0) / p1 (0,2,1)",
                                                                      "p0 (0,1,1) / p1 (0,1,1)",
                                                                      "p0 (0,1,2) / p1 (0,0,1)" } );
        CHECK( patchInterfaceEmbeddings( h1 ) == std::set<std::string>{ "0 1 | 0 0 | 1 1 / 1r 0 | 0 0 | 1 1" } );
    }

    SECTION( "2D row of three patches: connections in map order" )
    {
        const auto patch = makePatch( { two, two }, { 2, 2 } );
        const InternalConnectionsMap connections = {
            { ConstituentSide{ 0, 0 }, { TPPermutation::Flip1d, ConstituentSide{ 1, 1 } } },
            { ConstituentSide{ 1, 1 }, { TPPermutation::Flip1d, ConstituentSide{ 0, 0 } } },
            { ConstituentSide{ 1, 0 }, { TPPermutation::Flip1d, ConstituentSide{ 2, 1 } } },
            { ConstituentSide{ 2, 1 }, { TPPermutation::Flip1d, ConstituentSide{ 1, 0 } } } };
        const MultiPatchSplineSpace h1 = buildH1MultiPatchSplineSpace( { patch, patch, patch }, connections );
        CHECK( patchInterfaceOrder( h1 ) == std::vector<std::string>{ "p0 (1,0) / p1 (0,0)",
                                                                      "p0 (1,1) / p1 (0,1)",
                                                                      "p1 (1,0) / p2 (0,0)",
                                                                      "p1 (1,1) / p2 (0,1)" } );
        CHECK( patchInterfaceEmbeddings( h1 ) == std::set<std::string>{ "0 | 0 | 1 / 0 | 0 | 1" } );
    }
}

// Item 1, step 6: the fragment embedding formula of TraceMesh.hpp, and each
// rejection of a malformed embedding or record.
namespace
{
    TraceSideEmbedding embeddingOf( const std::vector<size_t>& source,
                                    const std::vector<bool>& reversed,
                                    const std::vector<double>& offset,
                                    const std::vector<double>& scale )
    {
        TraceSideEmbedding out;
        for( const size_t a : source ) out.axes.source_axis_for_destination.push_back( a );
        for( const bool r : reversed ) out.axes.source_axis_reversed.push_back( r );
        for( const double o : offset ) out.offset.push_back( o );
        for( const double c : scale ) out.scale.push_back( c );
        return out;
    }

    std::string rejection( const std::function<void()>& f )
    {
        try
        {
            f();
        }
        catch( const std::invalid_argument& e )
        {
            return e.what();
        }
        catch( const std::exception& e )
        {
            return std::string( "not invalid_argument: " ) + e.what();
        }
        return "accepted";
    }
}

TEST_CASE( "Trace side embeddings place fragment points by the documented formula" )
{
    // y_0 = 0.25 + 0.5 ( 1 - xi_1 ), y_1 = 0.5 + 0.25 xi_0.
    const Eigen::VectorXd y = embedTraceFragmentPoint(
        embeddingOf( { 1, 0 }, { true, false }, { 0.25, 0.5 }, { 0.5, 0.25 } ), Eigen::Vector2d( 0.2, 0.6 ) );
    REQUIRE( y.size() == 2 );
    CHECK_THAT( y( 0 ), Catch::Matchers::WithinAbs( 0.45, 1e-15 ) );
    CHECK_THAT( y( 1 ), Catch::Matchers::WithinAbs( 0.55, 1e-15 ) );

    // A reversed fragment on [2/3, 1] of its leaf side.
    const TraceSideEmbedding e = embeddingOf( { 0 }, { true }, { 2.0 / 3.0 }, { 1.0 / 3.0 } );
    CHECK_THAT( embedTraceFragmentPoint( e, Eigen::VectorXd::Constant( 1, 0.0 ) )( 0 ),
                Catch::Matchers::WithinAbs( 1.0, 1e-15 ) );
    CHECK_THAT( embedTraceFragmentPoint( e, Eigen::VectorXd::Constant( 1, 1.0 ) )( 0 ),
                Catch::Matchers::WithinAbs( 2.0 / 3.0, 1e-15 ) );

    CHECK( embeddingText( wholeTraceSideEmbedding( 1 ) ) == "0 | 0 | 1" );
    CHECK( embeddingText( wholeTraceSideEmbedding( 2 ) ) == "0 1 | 0 0 | 1 1" );
}

TEST_CASE( "Malformed trace side embeddings and records are rejected" )
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const auto embedding_rejection = []( const TraceSideEmbedding& e ) {
        return rejection( [&] { validateTraceSideEmbedding( e ); } );
    };
    const std::string outside = "Trace side embedding places the fragment outside its leaf side.";
    CHECK( embedding_rejection( embeddingOf( { 1, 0 }, { false, true }, { 0.0, 0.5 }, { 1.0, 0.5 } ) ) == "accepted" );
    CHECK( embedding_rejection( embeddingOf( { 0 }, { false }, { -1e-12 }, { 1.0 + 2e-12 } ) ) == "accepted" );
    CHECK( embedding_rejection( embeddingOf( { 0, 0 }, { false, false }, { 0, 0 }, { 1, 1 } ) ) ==
           "Side coordinate transform is not a permutation of the tangential axes." );
    CHECK( embedding_rejection( embeddingOf( {}, {}, {}, {} ) ) ==
           "Side coordinate transform must have rank one or two." );
    CHECK( embedding_rejection( embeddingOf( { 0, 1 }, { false }, { 0, 0 }, { 1, 1 } ) ) ==
           "Side coordinate transform axis and reversal ranks disagree." );
    CHECK( embedding_rejection( embeddingOf( { 0, 1 }, { false, false }, { 0 }, { 1, 1 } ) ) ==
           "Trace side embedding offset and scale must match the axis rank." );
    CHECK( embedding_rejection( embeddingOf( { 0 }, { false }, { nan }, { 1 } ) ) ==
           "Trace side embedding offset and scale must be finite." );
    CHECK( embedding_rejection( embeddingOf( { 0 }, { false }, { 0 }, { inf } ) ) ==
           "Trace side embedding offset and scale must be finite." );
    CHECK( embedding_rejection( embeddingOf( { 0 }, { false }, { 0 }, { 0 } ) ) ==
           "Trace side embedding scales must be positive." );
    CHECK( embedding_rejection( embeddingOf( { 0 }, { false }, { 1 }, { -0.5 } ) ) ==
           "Trace side embedding scales must be positive." );
    CHECK( embedding_rejection( embeddingOf( { 0 }, { false }, { -0.1 }, { 0.5 } ) ) == outside );
    CHECK( embedding_rejection( embeddingOf( { 0 }, { false }, { 0.6 }, { 0.5 } ) ) == outside );
    CHECK( rejection( [] { embedTraceFragmentPoint( embeddingOf( { 0 }, { false }, { 0.6 }, { 0.5 } ),
                                                    Eigen::VectorXd::Constant( 1, 0.5 ) ); } ) == outside );

    // s-max to s-max in 2D reverses t; s-max to s-min does not.
    const ElementSide s_max( 0, false );
    const ElementSide s_min( 0, true );
    const TraceSideEmbedding whole = wholeTraceSideEmbedding( 1 );
    const TraceSideEmbedding half = embeddingOf( { 0 }, { false }, { 0.5 }, { 0.5 } );
    const TraceSideEmbedding reversed = embeddingOf( { 0 }, { true }, { 0 }, { 1 } );
    const TraceSideEmbedding reversed_third = embeddingOf( { 0 }, { true }, { 2.0 / 3.0 }, { 1.0 / 3.0 } );
    const auto side = []( const ElementSide& s, const TraceSideEmbedding& e ) {
        return TraceMeshSide{ Cell( Dart( 0 ), 2 ), s, 0, 0, e };
    };
    const auto record_rejection = []( const TraceMeshInterface& iface ) {
        return rejection( [&] { validateTraceMeshInterface( iface ); } );
    };
    const auto boundary = [&]( const TraceSideEmbedding& e ) {
        return TraceMeshInterface{ TraceMeshEntityType::Boundary, side( s_max, e ), std::nullopt, std::nullopt };
    };
    const auto interior = [&]( const TraceSideEmbedding& a, const TraceSideEmbedding& b ) {
        return TraceMeshInterface{ TraceMeshEntityType::Interior, side( s_max, a ), side( s_min, b ), std::nullopt };
    };
    const auto patch = [&]( const ElementSide& second_side,
                            const TraceSideEmbedding& a,
                            const TraceSideEmbedding& b,
                            const std::optional<TPPermutation> permutation ) {
        return TraceMeshInterface{
            TraceMeshEntityType::PatchInterface, side( s_max, a ), side( second_side, b ), permutation };
    };
    const std::string not_whole = "Boundary and interior trace records must embed whole sides by the identity.";
    const std::string frame = "A patch interface fragment must use the first side's axes and the interface transform.";
    const std::string boundary_shape = "A boundary trace record must have no second side and no permutation.";
    const std::string interior_shape = "An interior trace record must have a second side and no permutation.";

    CHECK( record_rejection( boundary( whole ) ) == "accepted" );
    CHECK( record_rejection( interior( whole, whole ) ) == "accepted" );
    CHECK( record_rejection( patch( s_min, half, half, TPPermutation::Flip1d ) ) == "accepted" );
    CHECK( record_rejection( patch( s_max, whole, reversed_third, TPPermutation::Flip1d ) ) == "accepted" );

    CHECK( record_rejection( boundary( embeddingOf( { 0 }, { false }, { 0.6 }, { 0.5 } ) ) ) == outside );
    CHECK( record_rejection( interior( whole, wholeTraceSideEmbedding( 2 ) ) ) ==
           "Trace record sides have embeddings of different rank." );
    CHECK( record_rejection( boundary( half ) ) == not_whole );
    CHECK( record_rejection( boundary( reversed ) ) == not_whole );
    CHECK( record_rejection( interior( whole, half ) ) == not_whole );
    CHECK( record_rejection( TraceMeshInterface{
               TraceMeshEntityType::Boundary, side( s_max, whole ), side( s_min, whole ), std::nullopt } ) ==
           boundary_shape );
    CHECK( record_rejection( TraceMeshInterface{
               TraceMeshEntityType::Boundary, side( s_max, whole ), std::nullopt, TPPermutation::Flip1d } ) ==
           boundary_shape );
    CHECK( record_rejection( TraceMeshInterface{
               TraceMeshEntityType::Interior, side( s_max, whole ), std::nullopt, std::nullopt } ) == interior_shape );
    CHECK( record_rejection( TraceMeshInterface{
               TraceMeshEntityType::Interior, side( s_max, whole ), side( s_min, whole ), TPPermutation::Flip1d } ) ==
           interior_shape );
    CHECK( record_rejection( patch( s_min, whole, whole, std::nullopt ) ) ==
           "A patch interface trace record needs a second side and a permutation." );
    CHECK( record_rejection( TraceMeshInterface{
               TraceMeshEntityType::PatchInterface, side( s_max, whole ), std::nullopt, TPPermutation::Flip1d } ) ==
           "A patch interface trace record needs a second side and a permutation." );
    CHECK( record_rejection( patch( s_max, whole, whole, TPPermutation::Flip1d ) ) == frame );
    CHECK( record_rejection( patch( s_min, whole, reversed, TPPermutation::Flip1d ) ) == frame );
    CHECK( record_rejection( patch( s_max, reversed, reversed, TPPermutation::Flip1d ) ) == frame );
}

// Item 1, step 6 (outstanding_issues.txt item 16): a leaf whose faces finer
// leaves subdivide holds own-level darts that are not leaf darts.  Every
// record must still name its element by a leaf dart.
namespace
{
    using Leaf3d = std::array<size_t, 5>; // level, patch, element indices

    // Two 2x2x2 patches, patch 0's s-max side glued to patch 1's s-min side.
    // Patch 0 keeps (1,0,0) coarse and refines the rest; patch 1 refines only
    // (0,0,0).  So finer leaves split the s-min, t-max and u-max faces of
    // p0 (1,0,0) inside patch 0 and its interface face from patch 1.
    std::vector<Leaf3d> splitFaceLeaves()
    {
        std::vector<Leaf3d> out;
        for( size_t patch = 0; patch < 2; patch++ )
            for( size_t i = 0; i < 2; i++ )
                for( size_t j = 0; j < 2; j++ )
                    for( size_t k = 0; k < 2; k++ )
                    {
                        const bool corner = j == 0 and k == 0 and i == ( patch == 0 ? 1 : 0 );
                        if( corner == ( patch == 0 ) )
                        {
                            out.push_back( { 0, patch, i, j, k } );
                            continue;
                        }
                        for( size_t a = 0; a < 2; a++ )
                            for( size_t b = 0; b < 2; b++ )
                                for( size_t c = 0; c < 2; c++ )
                                    out.push_back( { 1, patch, 2 * i + a, 2 * j + b, 2 * k + c } );
                    }
        return out;
    }

    HierarchicalMultiPatchSplineSpace makeSplitFaceHierarchy( const std::vector<Leaf3d>& leaves )
    {
        const double ptol = 1e-10;
        const InternalConnectionsMap connections =
            twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ), TPPermutation::ZeroToZero );
        std::vector<std::shared_ptr<const MultiPatchSplineSpace>> levels;
        for( const KnotVector& kv : { KnotVector( { 0, 0, 0, 0.5, 1, 1, 1 }, ptol ),
                                      KnotVector( { 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1 }, ptol ) } )
        {
            const auto patch = makePatch( { kv, kv, kv }, { 2, 2, 2 } );
            levels.push_back( std::make_shared<const MultiPatchSplineSpace>(
                buildH1MultiPatchSplineSpace( { patch, patch }, connections ) ) );
        }
        std::vector<std::vector<Cell>> cells( levels.size() );
        for( const Leaf3d& l : leaves )
        {
            const auto& cmap = dynamic_cast<const MultiPatchCombinatorialMap&>(
                levels.at( l[0] )->basisComplex().parametricAtlas().cmap() );
            cells.at( l[0] ).push_back( patchTopCellAt( cmap, l[1], { l[2], l[3], l[4] } ) );
        }
        return buildHierarchicalSplineSpace( levels, cells );
    }

    std::string leafSideText( const Leaf3d& l, const ElementSide& side )
    {
        return "L" + std::to_string( l[0] ) + " p" + std::to_string( l[1] ) + " (" + std::to_string( l[2] ) + "," +
               std::to_string( l[3] ) + "," + std::to_string( l[4] ) + ") side " + std::to_string( side.sideId() );
    }

    // Interior pairs carry no order; patch pairs keep the enumeration's.
    std::string pairText( const std::string& first, const std::string& second )
    {
        return first + " / " + second;
    }

    // Throws unless the element is named by a leaf dart.
    Leaf3d leafOf( const HierarchicalMultiPatchCombinatorialMap& cmap, const Cell& element )
    {
        const auto [level, level_d] = cmap.unrefinedAncestorDartOfCell( element );
        const MultiPatchCombinatorialMap& level_cmap = *cmap.refinementLevels().at( level );
        const auto [patch, patch_d] = level_cmap.toLocalDart( level_d );
        const FullyUnflattenedDart unflat = unflattenFull( *level_cmap.constituents().at( patch ), patch_d );
        return { level,
                 patch,
                 unflat.unflat_darts.at( 0 ).id(),
                 unflat.unflat_darts.at( 1 ).id(),
                 unflat.unflat_darts.at( 2 ).id() };
    }

    // The records an independent enumeration of the leaves predicts.
    struct SplitFaceRecords
    {
        std::vector<std::string> boundary;
        std::vector<std::string> interior;
        std::vector<std::string> patch;
    };

    SplitFaceRecords expectedSplitFaceRecords( const std::vector<Leaf3d>& leaves )
    {
        const std::set<Leaf3d> leaf_set( leaves.begin(), leaves.end() );
        const auto per_axis = []( const Leaf3d& l ) { return size_t( 2 ) << l[0]; };
        const auto extent = [&]( const Leaf3d& l, const size_t axis ) {
            const double h = 1.0 / double( per_axis( l ) );
            return std::array<double, 2>{ h * double( l[2 + axis] ), h * double( l[2 + axis] + 1 ) };
        };

        SplitFaceRecords out;
        for( const Leaf3d& l : leaves )
            for( size_t axis = 0; axis < 3; axis++ )
            {
                const ElementSide lower( axis, true );
                const ElementSide upper( axis, false );
                if( l[2 + axis] == 0 and not( axis == 0 and l[1] == 1 ) )
                    out.boundary.push_back( leafSideText( l, lower ) );
                if( l[2 + axis] + 1 == per_axis( l ) and not( axis == 0 and l[1] == 0 ) )
                    out.boundary.push_back( leafSideText( l, upper ) );
                Leaf3d next = l;
                next[2 + axis]++;
                if( leaf_set.contains( next ) )
                    out.interior.push_back( pairText( leafSideText( l, upper ), leafSideText( next, lower ) ) );
            }
        for( const Leaf3d& a : leaves )
            for( const Leaf3d& b : leaves )
            {
                if( a[1] != 0 or b[1] != 1 or a[2] + 1 != per_axis( a ) or b[2] != 0 ) continue;
                bool overlap = true;
                for( size_t axis = 1; axis < 3; axis++ )
                    overlap = overlap and std::min( extent( a, axis )[1], extent( b, axis )[1] ) >
                                              std::max( extent( a, axis )[0], extent( b, axis )[0] ) + 1e-12;
                if( overlap )
                    out.patch.push_back( pairText( leafSideText( a, ElementSide( 0, false ) ),
                                                   leafSideText( b, ElementSide( 0, true ) ) ) );
            }
        for( std::vector<std::string>* v : { &out.boundary, &out.interior, &out.patch } )
            std::sort( v->begin(), v->end() );
        return out;
    }

    struct RecordProblems
    {
        std::vector<std::string> records;     // a side not named by a leaf dart, or without trace data
        std::vector<std::string> difference;  // missing or unexpected against the prediction
    };

    RecordProblems recordProblems( const SplineSpace& ss,
                                   const std::string& kind,
                                   const std::vector<TraceMeshInterface>& records,
                                   const std::set<Leaf3d>& leaves,
                                   const std::vector<std::string>& expected )
    {
        const auto& cmap =
            dynamic_cast<const HierarchicalMultiPatchCombinatorialMap&>( ss.basisComplex().parametricAtlas().cmap() );
        RecordProblems out;
        const auto named = [&]( const TraceMeshSide& side ) {
            try
            {
                const Leaf3d l = leafOf( cmap, side.element );
                if( l[0] != side.level or l[1] != side.patch_id or not leaves.contains( l ) )
                    out.records.push_back( kind + " record names " + leafSideText( l, side.side ) );
                return leafSideText( l, side.side );
            }
            catch( const std::exception& e )
            {
                out.records.push_back( kind + " record side " + std::to_string( side.side.sideId() ) + " in patch " +
                                       std::to_string( side.patch_id ) + " at level " + std::to_string( side.level ) +
                                       ": " + e.what() );
                return std::string( "?" );
            }
        };

        std::vector<std::string> found;
        for( const TraceMeshInterface& iface : records )
        {
            std::string text = named( iface.first );
            if( iface.second.has_value() ) text = pairText( text, named( *iface.second ) );
            found.push_back( text );
            try
            {
                const TraceInterfaceElement trace = traceInterfaceElement( ss, iface );
                if( trace.first.connectivity.empty() or
                    ( trace.second.has_value() and trace.second->connectivity.empty() ) )
                    out.records.push_back( kind + " record " + text + " has an empty trace" );
            }
            catch( const std::exception& e )
            {
                out.records.push_back( kind + " record " + text + " has no trace data: " + e.what() );
            }
        }
        std::sort( found.begin(), found.end() );
        std::vector<std::string> missing;
        std::vector<std::string> extra;
        std::set_difference(
            expected.begin(), expected.end(), found.begin(), found.end(), std::back_inserter( missing ) );
        std::set_difference(
            found.begin(), found.end(), expected.begin(), expected.end(), std::back_inserter( extra ) );
        for( const std::string& m : missing ) out.difference.push_back( kind + " record missing: " + m );
        for( const std::string& e : extra ) out.difference.push_back( kind + " record unexpected: " + e );
        return out;
    }

    std::string lines( const std::vector<std::string>& v )
    {
        std::string out;
        for( const std::string& s : v ) out += s + "\n";
        return out;
    }

    struct SplitFaceSpaces
    {
        HierarchicalMultiPatchSplineSpace h1;
        VectorConformingHierarchicalMultiPatchSplineSpace hdiv;
        VectorConformingHierarchicalMultiPatchSplineSpace hcurl;

        std::vector<std::pair<std::string, const SplineSpace*>> named() const
        {
            return { { "H1", &h1 }, { "H(div)", &hdiv }, { "H(curl)", &hcurl } };
        }
    };

    SplitFaceSpaces makeSplitFaceSpaces( const std::vector<Leaf3d>& leaves )
    {
        const HierarchicalMultiPatchSplineSpace h1 = makeSplitFaceHierarchy( leaves );
        return { h1,
                 buildHDivHierarchicalMultiPatchSplineSpace( h1 ),
                 buildHCurlHierarchicalMultiPatchSplineSpace( h1 ) };
    }
}

TEST_CASE( "3D trace records name a leaf with subdivided faces by a leaf dart" )
{
    const std::vector<Leaf3d> leaves = splitFaceLeaves();
    const std::set<Leaf3d> leaf_set( leaves.begin(), leaves.end() );
    const SplitFaceRecords expected = expectedSplitFaceRecords( leaves );
    // Boundary area 10: 20 level-0 faces of area 1/4 and 80 level-1 faces of area 1/16.
    CHECK( expected.boundary.size() == 100 );
    CHECK( expected.patch.size() == 16 );
    // Same-level neighbours: 9 in patch 1 at level 0; 120 in patch 0 and 12 in patch 1 at level 1.
    CHECK( expected.interior.size() == 141 );

    const SplitFaceSpaces spaces = makeSplitFaceSpaces( leaves );
    for( const auto& [name, ss] : spaces.named() )
    {
        INFO( name );
        const RecordProblems boundary =
            recordProblems( *ss, "boundary", boundaryTraceMeshInterfaces( *ss ), leaf_set, expected.boundary );
        const RecordProblems interior =
            recordProblems( *ss, "interior", interiorTraceMeshInterfaces( *ss ), leaf_set, expected.interior );
        const RecordProblems patch =
            recordProblems( *ss, "patch", patchTraceMeshInterfaces( *ss ), leaf_set, expected.patch );
        CHECK( lines( boundary.records ) + lines( interior.records ) + lines( patch.records ) == "" );
        CHECK( lines( boundary.difference ) + lines( interior.difference ) + lines( patch.difference ) == "" );
    }
}
