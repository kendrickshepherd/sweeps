// Item 1, step 10.4 (outstanding_issues.txt): the leaf and support queries an adaptive loop reads
// from a single-patch hierarchy, each against an oracle built here from the marks or the knots.
#include <catch2/catch_test_macros.hpp>
#include <NavierStokesDiscretization.hpp>
#include <HierarchicalTPCombinatorialMap.hpp>
#include <CombinatorialMapMethods.hpp>
#include <KnotVector.hpp>
#include <TPSplineSpace.hpp>
#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace basis;
using namespace topology;

namespace
{
    using Index = std::vector<size_t>;

    struct Layout
    {
        std::string name;
        std::vector<double> breaks_s;
        std::vector<double> breaks_t;
        std::vector<std::vector<std::pair<size_t, size_t>>> marks;
    };

    // "Three levels, jump 2 inside a patch" (HierarchicalTPSplineSpace_test) with t halved per level too:
    // leaves L0 [0,1], L1 [1.5,2] and L2 [1,1.5] in s, each across t in [0,1].
    const Layout jump_two_across_face{
        "jump 2 across a face", { 0, 1, 2 }, { 0, 1 }, { { { 1, 0 } }, { { 2, 0 }, { 2, 1 } } } };

    // Edge graded: the level 2 leaves in [1.5,2]^2 meet the level 0 leaf [2,3]^2 only at the point (2,2).
    const Layout jump_two_at_corner{ "jump 2 at a corner only",
                                     { 0, 1, 2, 3, 4 },
                                     { 0, 1, 2, 3, 4 },
                                     { { { 1, 1 }, { 2, 1 }, { 1, 2 } }, { { 3, 3 } } } };

    const std::vector<Layout> layouts = { jump_two_across_face, jump_two_at_corner };
    const std::vector<size_t> degrees = { 2, 3 };

    std::string describe( const Index& ix )
    {
        std::string out = "(";
        for( size_t i = 0; i < ix.size(); i++ )
            out += ( i == 0 ? "" : i == 1 ? "; " : ", " ) + std::to_string( ix.at( i ) );
        return out + ")";
    }

    std::string join( const std::vector<std::string>& problems )
    {
        std::string out;
        for( const std::string& p : problems ) out += p + "\n";
        return out;
    }

    std::vector<double> levelBreaks( std::vector<double> breaks, const size_t level )
    {
        for( size_t l = 0; l < level; l++ )
        {
            std::vector<double> finer = { breaks.front() };
            for( size_t i = 1; i < breaks.size(); i++ )
                finer.insert( finer.end(), { 0.5 * ( breaks.at( i - 1 ) + breaks.at( i ) ), breaks.at( i ) } );
            breaks = std::move( finer );
        }
        return breaks;
    }

    // Simple interior knots, end knots of multiplicity degree + 1.
    std::vector<double> openKnots( const std::vector<double>& breaks, const size_t degree )
    {
        std::vector<double> knots( degree, breaks.front() );
        knots.insert( knots.end(), breaks.begin(), breaks.end() );
        knots.insert( knots.end(), degree, breaks.back() );
        return knots;
    }

    // Per function k of the degree q basis on knots: the spans of breaks inside [knots_k, knots_(k+q+1)].
    std::vector<std::vector<size_t>> supportSpans( const std::vector<double>& knots,
                                                   const std::vector<double>& breaks,
                                                   const size_t q )
    {
        std::vector<std::vector<size_t>> out( knots.size() - q - 1 );
        for( size_t k = 0; k < out.size(); k++ )
            for( size_t e = 0; e + 1 < breaks.size(); e++ )
                if( knots.at( k ) <= breaks.at( e ) + 1e-12 and breaks.at( e + 1 ) <= knots.at( k + q + 1 ) + 1e-12 )
                    out.at( k ).push_back( e );
        return out;
    }

    // Sorted cell supports of the tensor-product functions, function id k_s + n_s * ( k_t + n_t * k_u ).
    std::vector<std::vector<Index>> tensorSupports( const std::vector<std::vector<std::vector<size_t>>>& spans )
    {
        std::vector<std::vector<Index>> out( 1, std::vector<Index>( 1 ) );
        for( const std::vector<std::vector<size_t>>& direction : spans )
        {
            std::vector<std::vector<Index>> next;
            for( const std::vector<size_t>& k_spans : direction )
                for( const std::vector<Index>& cells : out )
                {
                    std::vector<Index> product;
                    for( const Index& cell : cells )
                        for( const size_t e : k_spans )
                        {
                            product.push_back( cell );
                            product.back().push_back( e );
                        }
                    std::sort( product.begin(), product.end() );
                    next.push_back( product );
                }
            out = std::move( next );
        }
        return out;
    }

    // The degree q supports on one level of a 2d layout.
    std::vector<std::vector<Index>> levelSupports( const Layout& layout, const size_t level, const size_t q )
    {
        const std::vector<double> breaks_s = levelBreaks( layout.breaks_s, level );
        const std::vector<double> breaks_t = levelBreaks( layout.breaks_t, level );
        return tensorSupports( { supportSpans( openKnots( breaks_s, q ), breaks_s, q ),
                                 supportSpans( openKnots( breaks_t, q ), breaks_t, q ) } );
    }

    // (level, indices) of every leaf: all level 0 cells, each mark replaced by its children 2 i + {0, 1}.
    std::set<Index> leavesFromMarks( const Index& level0_extents, const std::vector<std::vector<Index>>& marks )
    {
        std::set<Index> leaves = { { 0 } };
        for( const size_t n : level0_extents )
        {
            std::set<Index> next;
            for( const Index& ix : leaves )
                for( size_t i = 0; i < n; i++ )
                {
                    Index grown = ix;
                    grown.push_back( i );
                    next.insert( grown );
                }
            leaves = std::move( next );
        }
        for( size_t level = 0; level < marks.size(); level++ )
            for( const Index& m : marks.at( level ) )
            {
                Index key = { level };
                key.insert( key.end(), m.begin(), m.end() );
                if( leaves.erase( key ) != 1 ) throw std::runtime_error( "mark " + describe( key ) + " is not a leaf" );
                for( size_t child = 0; child < ( size_t{ 1 } << m.size() ); child++ )
                {
                    Index c = { level + 1 };
                    for( size_t i = 0; i < m.size(); i++ ) c.push_back( 2 * m.at( i ) + ( ( child >> i ) & 1 ) );
                    leaves.insert( c );
                }
            }
        return leaves;
    }

    std::vector<std::vector<Index>> marksOf( const Layout& layout )
    {
        std::vector<std::vector<Index>> out;
        for( const auto& level_marks : layout.marks )
        {
            out.emplace_back();
            for( const auto& [s, t] : level_marks ) out.back().push_back( { s, t } );
        }
        return out;
    }

    std::unique_ptr<const api::NavierStokesHierarchicalDiscretization> build( const Layout& layout,
                                                                             const size_t degree )
    {
        const KnotVector kv_s( openKnots( layout.breaks_s, degree ), 1e-10 );
        const KnotVector kv_t( openKnots( layout.breaks_t, degree ), 1e-10 );
        const Eigen::MatrixXd cpts = grevillePoints( buildBSpline( { kv_s, kv_t }, { degree, degree } ) ).transpose();
        return std::make_unique<const api::NavierStokesHierarchicalDiscretization>(
            kv_s, kv_t, degree, degree, cpts, layout.marks );
    }

    Index keyOf( const LeafElementInfo& info )
    {
        Index key = { info.level };
        key.insert( key.end(), info.indices.begin(), info.indices.end() );
        return key;
    }

    // leafElementInfos against the marks, and leafElementInfo from every dart of every leaf.
    std::vector<std::string> leafInfoProblems( const std::string& prefix,
                                               const HierarchicalTPCombinatorialMap& cmap,
                                               const std::set<Index>& expected )
    {
        std::vector<std::string> problems;
        std::multiset<Index> actual;
        size_t other_level_darts = 0;
        for( const LeafElementInfo& info : leafElementInfos( cmap ) )
        {
            const Index key = keyOf( info );
            actual.insert( key );
            if( info.indices.size() != cmap.dim() )
                problems.push_back( prefix + describe( key ) + " has the wrong length" );
            iterateDartsOfCell( cmap, info.leaf, [&]( const Dart& d ) {
                if( cmap.dartRanges().toLocalDart( d ).first != info.level ) other_level_darts++;
                const Index by_dart = keyOf( leafElementInfo( cmap, Cell( d, cmap.dim() ) ) );
                if( by_dart != key )
                    problems.push_back( prefix + "dart " + std::to_string( d.id() ) + " of " + describe( key ) +
                                        " gives " + describe( by_dart ) );
                return true;
            } );
        }
        if( actual != std::multiset<Index>( expected.begin(), expected.end() ) )
        {
            std::string diff;
            for( const Index& k : actual )
                if( not expected.contains( k ) ) diff += " extra " + describe( k );
            for( const Index& k : expected )
                if( actual.count( k ) != 1 ) diff += " missing or repeated " + describe( k );
            problems.push_back( prefix + "leaves differ from the marks:" + diff );
        }
        if( other_level_darts == 0 ) problems.push_back( prefix + "no leaf has a dart of another level" );
        return problems;
    }

    std::vector<std::string> supportProblems( const std::string& prefix,
                                              const TPSplineSpace& ss,
                                              const std::vector<std::vector<Index>>& expected )
    {
        const std::vector<std::vector<SmallVector<size_t, 3>>> actual = tensorCellSupports( ss );
        if( ss.numFunctions() != expected.size() or actual.size() != expected.size() )
            return { prefix + std::to_string( actual.size() ) + " supports of " + std::to_string( ss.numFunctions() ) +
                     " functions; the knot oracle has " + std::to_string( expected.size() ) };

        std::vector<std::string> problems;
        for( size_t fid = 0; fid < expected.size(); fid++ )
        {
            std::vector<Index> cells;
            for( const auto& c : actual.at( fid ) ) cells.emplace_back( c.begin(), c.end() );
            std::sort( cells.begin(), cells.end() );
            if( cells != expected.at( fid ) )
                problems.push_back( prefix + "function " + std::to_string( fid ) + " has " +
                                    std::to_string( cells.size() ) + " cells, the knot oracle " +
                                    std::to_string( expected.at( fid ).size() ) );
        }
        return problems;
    }

    // L2 active functions against: supp in Omega^l and not in Omega^(l+1), on the knot oracle supports.
    std::vector<std::string> activeProblems( const std::string& prefix,
                                             const api::NavierStokesHierarchicalDiscretization& nsd,
                                             const Layout& layout,
                                             const size_t degree )
    {
        std::vector<std::string> problems;
        const size_t n_levels = layout.marks.size() + 1;
        if( nsd.L2_ss.activeFunctions().size() != n_levels )
            return { prefix + std::to_string( nsd.L2_ss.activeFunctions().size() ) + " levels of active functions" };

        std::vector<std::set<std::pair<size_t, size_t>>> marked( n_levels );
        for( size_t level = 0; level < layout.marks.size(); level++ )
            marked.at( level ).insert( layout.marks.at( level ).begin(), layout.marks.at( level ).end() );

        for( size_t level = 0; level < n_levels; level++ )
        {
            const auto in_omega = [&]( const Index& cell ) {
                return level == 0 or marked.at( level - 1 ).contains( { cell.at( 0 ) / 2, cell.at( 1 ) / 2 } );
            };
            const auto in_finer_omega = [&]( const Index& cell ) {
                return marked.at( level ).contains( { cell.at( 0 ), cell.at( 1 ) } );
            };

            std::vector<size_t> expected;
            const std::vector<std::vector<Index>> supports = levelSupports( layout, level, degree - 1 );
            for( size_t fid = 0; fid < supports.size(); fid++ )
                if( std::all_of( supports.at( fid ).begin(), supports.at( fid ).end(), in_omega ) and
                    not std::all_of( supports.at( fid ).begin(), supports.at( fid ).end(), in_finer_omega ) )
                    expected.push_back( fid );

            std::vector<size_t> actual;
            for( const FunctionId& fid : nsd.L2_ss.activeFunctions().at( level ) ) actual.push_back( fid.id() );
            std::sort( actual.begin(), actual.end() );

            if( actual != expected )
            {
                std::string diff;
                for( const size_t f : actual )
                    if( not std::binary_search( expected.begin(), expected.end(), f ) )
                        diff += " extra " + std::to_string( f );
                for( const size_t f : expected )
                    if( not std::binary_search( actual.begin(), actual.end(), f ) )
                        diff += " missing " + std::to_string( f );
                problems.push_back( prefix + "level " + std::to_string( level ) + " active L2 functions differ:" +
                                    diff );
            }
        }
        return problems;
    }

    template <typename Check>
    void checkEachFixture( const Check& check )
    {
        for( const Layout& layout : layouts )
            for( const size_t degree : degrees )
            {
                const std::string prefix = layout.name + ", degree " + std::to_string( degree ) + ": ";
                std::vector<std::string> problems;
                try
                {
                    problems = check( prefix, layout, degree );
                }
                catch( const std::exception& e )
                {
                    problems = { prefix + "throws: " + e.what() };
                }
                INFO( join( problems ) );
                CHECK( problems.empty() );
            }
    }
}

TEST_CASE( "Leaf element info: level-local tensor indices from any dart, against the marks", "[amr_queries]" )
{
    checkEachFixture( []( const std::string& prefix, const Layout& layout, const size_t degree ) {
        const auto nsd = build( layout, degree );
        const Index extents = { layout.breaks_s.size() - 1, layout.breaks_t.size() - 1 };
        return leafInfoProblems(
            prefix, nsd->H1_ss.basisComplex().parametricAtlas().cmap(), leavesFromMarks( extents, marksOf( layout ) ) );
    } );
}

TEST_CASE( "Leaf element info refuses a cell that is not top-dimensional", "[amr_queries]" )
{
    const auto nsd = build( jump_two_across_face, 2 );
    const HierarchicalTPCombinatorialMap& cmap = nsd->H1_ss.basisComplex().parametricAtlas().cmap();
    const Cell leaf = leafElementInfos( cmap ).front().leaf;
    CHECK_NOTHROW( leafElementInfo( cmap, leaf ) );
    for( const uint dim : { 0u, 1u, 3u } )
    {
        CAPTURE( dim );
        CHECK_THROWS_AS( leafElementInfo( cmap, Cell( leaf.dart(), dim ) ), std::invalid_argument );
    }
}

TEST_CASE( "Leaf element info in 3d: level-local tensor indices from any dart, against the marks", "[amr_queries]" )
{
    // 2 x 3 x 2 level 0 cells; (0, 1, 0) refined, then its child (1, 3, 0), so the leaves change under s <-> u.
    const Index extents = { 2, 3, 2 };
    const std::vector<std::vector<Index>> marks = { { { 0, 1, 0 } }, { { 1, 3, 0 } } };
    std::vector<std::shared_ptr<const TPCombinatorialMap>> levels;
    for( size_t level = 0; level <= marks.size(); level++ )
    {
        const auto line = [&]( const size_t i ) {
            return std::make_shared<const CombinatorialMap1d>( extents.at( i ) << level );
        };
        levels.push_back( std::make_shared<const TPCombinatorialMap>(
            std::make_shared<const TPCombinatorialMap>( line( 0 ), line( 1 ) ), line( 2 ) ) );
    }

    std::vector<std::string> problems;
    try
    {
        const std::set<Index> expected = leavesFromMarks( extents, marks );
        std::vector<std::vector<Cell>> leaves( levels.size() );
        for( const Index& k : expected )
            leaves.at( k.at( 0 ) ).emplace_back(
                flattenFull( *levels.at( k.at( 0 ) ),
                             FullyUnflattenedDart( { Dart( k.at( 1 ) ), Dart( k.at( 2 ) ), Dart( k.at( 3 ) ) } ) ),
                3 );
        const HierarchicalTPCombinatorialMap cmap( levels, leaves );
        problems = leafInfoProblems( "3d: ", cmap, expected );
    }
    catch( const std::exception& e )
    {
        problems = { std::string( "3d: throws: " ) + e.what() };
    }
    INFO( join( problems ) );
    CHECK( problems.empty() );
}

TEST_CASE( "Tensor cell supports of the L2 level spaces, against the level knots at degree p - 1", "[amr_queries]" )
{
    checkEachFixture( []( const std::string& prefix, const Layout& layout, const size_t degree ) {
        const auto nsd = build( layout, degree );
        const HierarchicalTPCombinatorialMap& cmap = nsd->H1_ss.basisComplex().parametricAtlas().cmap();
        std::vector<std::string> problems;
        if( &nsd->L2_ss.basisComplex().parametricAtlas().cmap() != &cmap )
            problems.push_back( prefix + "L2 and H1 do not share the hierarchical cmap" );
        if( nsd->L2_ss.refinementLevels().size() != layout.marks.size() + 1 )
            return std::vector<std::string>{ prefix + "wrong number of L2 levels" };
        for( size_t level = 0; level < nsd->L2_ss.refinementLevels().size(); level++ )
        {
            const TPSplineSpace& level_ss = *nsd->L2_ss.refinementLevels().at( level );
            if( &level_ss.basisComplex().parametricAtlas().cmap() != cmap.refinementLevels().at( level ).get() )
                problems.push_back( prefix + "L2 level " + std::to_string( level ) + " has its own cmap" );
            const auto level_problems = supportProblems( prefix + "level " + std::to_string( level ) + ", ",
                                                         level_ss,
                                                         levelSupports( layout, level, degree - 1 ) );
            problems.insert( problems.end(), level_problems.begin(), level_problems.end() );
        }
        return problems;
    } );
}

TEST_CASE( "Tensor cell supports in 3d, against the knots", "[amr_queries]" )
{
    const std::vector<std::vector<double>> breaks = { { 0, 1, 2, 3 }, { 0, 0.5, 2 }, { 0, 1 } };
    const std::vector<size_t> q = { 2, 1, 3 };
    SmallVector<KnotVector, 3> kvs;
    std::vector<std::vector<std::vector<size_t>>> spans;
    for( size_t i = 0; i < 3; i++ )
    {
        kvs.push_back( KnotVector( openKnots( breaks.at( i ), q.at( i ) ), 1e-10 ) );
        spans.push_back( supportSpans( openKnots( breaks.at( i ), q.at( i ) ), breaks.at( i ), q.at( i ) ) );
    }

    std::vector<std::string> problems;
    try
    {
        problems = supportProblems(
            "3d: ", buildBSpline( kvs, { q.at( 0 ), q.at( 1 ), q.at( 2 ) } ), tensorSupports( spans ) );
    }
    catch( const std::exception& e )
    {
        problems = { std::string( "3d: throws: " ) + e.what() };
    }
    INFO( join( problems ) );
    CHECK( problems.empty() );
}

TEST_CASE( "Active L2 functions follow support containment", "[amr_queries]" )
{
    checkEachFixture( []( const std::string& prefix, const Layout& layout, const size_t degree ) {
        return activeProblems( prefix, *build( layout, degree ), layout, degree );
    } );
}
