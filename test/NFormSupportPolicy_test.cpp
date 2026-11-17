// Item 1, step 10.6 (outstanding_issues.txt): the n-form support policy P_n and the check / require api, each
// verdict against an oracle built here from the marks and the level knots, and against a value derived by hand.
#include <catch2/catch_test_macros.hpp>
#include <RefinementPolicy.hpp>
#include <NFormSupportPolicy.hpp>
#include <HierarchicalMultiPatchCombinatorialMap.hpp>
#include <HierarchicalMultiPatchSplineSpace.hpp>
#include <HierarchicalTPSplineSpace.hpp>
#include <CombinatorialMapMethods.hpp>
#include <KnotVector.hpp>
#include <TPSplineSpace.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace basis;
using namespace topology;

namespace
{
    using Index = std::vector<size_t>;
    using Marks = std::vector<std::vector<api::PatchElement>>;
    using api::PatchSide;
    using api::RefinementPolicy;

    // x = x0 + xs s + xt t, y = y0 + ys s + yt t.
    struct AxisMap
    {
        double x0, xs, xt, y0, ys, yt;
    };

    struct PatchLayout
    {
        std::vector<double> breaks_s;
        std::vector<double> breaks_t;
        AxisMap phi;
    };

    struct Failure
    {
        size_t patch;
        size_t level;
        Index cell;

        bool operator==( const Failure& ) const = default;
    };

    using Verdict = std::optional<Failure>;

    struct Layout
    {
        std::string name;
        std::vector<PatchLayout> patches;
        std::vector<std::pair<api::PatchSideRef, api::PatchSideRef>> gluing;
        Marks marks;
        std::map<size_t, Verdict> by_hand; // per degree
        std::set<size_t> interior_levels;  // levels whose marks every meeting support covers whole
    };

    const AxisMap identity{ 0, 1, 0, 0, 0, 1 };

    // The step 10.5 layouts (test/HierarchicalMultiPatchAmr_test.cpp); verdicts from the oracle only.
    const Layout side_by_side{ "side by side, S1 to S0",
                               { { { 0, 1, 2 }, { 0, 1, 3 }, identity },
                                 { { 0, 1, 2, 3 }, { 0, 1, 3 }, { 2, 1, 0, 0, 0, 1 } } },
                               { { { 0, PatchSide::S1 }, { 1, PatchSide::S0 } } },
                               { { { 0, 1, 0 } }, { { 0, 3, 0 }, { 0, 3, 1 } } },
                               {},
                               {} };
    const Layout stacked{ "stacked, T1 to T0, translated s",
                          { { { 0, 1, 2 }, { 0, 1 }, identity }, { { 5, 6, 7 }, { 0, 2 }, { -5, 1, 0, 1, 0, 1 } } },
                          { { { 0, PatchSide::T1 }, { 1, PatchSide::T0 } } },
                          { { { 1, 1, 0 } }, { { 1, 2, 0 } } },
                          {},
                          {} };
    const Layout half_turn{ "half turn, S1 to S1",
                            { { { 0, 1, 2 }, { 0, 1, 3 }, identity }, { { 0, 1, 2 }, { 0, 2, 3 }, { 4, -1, 0, 3, 0, -1 } } },
                            { { { 0, PatchSide::S1 }, { 1, PatchSide::S1 } } },
                            { { { 0, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 } }, { { 1, 3, 1 } } },
                            {},
                            {} };
    const Layout quarter_turn{ "quarter turn, S1 to T0",
                               { { { 0, 1, 2 }, { 0, 1, 3 }, identity }, { { 0, 2, 3 }, { 0, 1, 2 }, { 2, 0, 1, 3, -1, 0 } } },
                               { { { 0, PatchSide::S1 }, { 1, PatchSide::T0 } } },
                               { { { 1, 1, 0 }, { 0, 1, 1 } } },
                               {},
                               {} };

    const std::vector<double> eight = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    const PatchLayout square{ eight, eight, identity };
    const PatchLayout right_square{ eight, eight, { 8, 1, 0, 0, 0, 1 } };
    const PatchLayout right_square_turned{ eight, eight, { 16, -1, 0, 8, 0, -1 } };
    const std::vector<size_t> fixture_degrees = { 2, 3 };

    Verdict fails( const size_t patch, const size_t level, const Index& cell ) { return Failure{ patch, level, cell }; }

    std::map<size_t, Verdict> both( const Verdict& v ) { return { { 2, v }, { 3, v } }; }

    std::vector<api::PatchElement> block( const size_t patch, const size_t lo, const size_t hi )
    {
        std::vector<api::PatchElement> out;
        for( size_t j = lo; j <= hi; j++ )
            for( size_t i = lo; i <= hi; i++ ) out.push_back( { patch, i, j } );
        return out;
    }

    // q = p - 1 is the L2 degree: interior L2 supports are p cells wide, boundary ones shorter.
    const Layout corner{ "corner cell", { square }, {}, { { { 0, 0, 0 } } }, both( std::nullopt ), {} };
    const Layout edge{ "edge cell", { square }, {}, { { { 0, 3, 0 } } }, both( fails( 0, 0, { 3, 0 } ) ), {} };
    const Layout interior{ "interior cell", { square }, {}, { { { 0, 3, 3 } } }, both( fails( 0, 0, { 3, 3 } ) ), { 0 } };
    const Layout block2{
        "interior 2 x 2 block", { square }, {}, { block( 0, 3, 4 ) }, { { 2, std::nullopt }, { 3, fails( 0, 0, { 3, 3 } ) } }, { 0 } };
    const Layout block3{ "interior 3 x 3 block", { square }, {}, { block( 0, 3, 5 ) }, both( std::nullopt ), { 0 } };
    const Layout antidiagonal{ "two interior cells on an antidiagonal",
                               { square },
                               {},
                               { { { 0, 5, 2 }, { 0, 2, 5 } } },
                               both( fails( 0, 0, { 5, 2 } ) ),
                               { 0 } };
    const Layout three_levels{ "three levels, failing on the last checked level",
                               { square },
                               {},
                               { block( 0, 3, 5 ), { { 0, 8, 8 } } },
                               both( fails( 0, 1, { 8, 8 } ) ),
                               { 0, 1 } };
    const std::vector<Layout> one_patch = { corner, edge, interior, block2, block3, antidiagonal, three_levels };

    const std::vector<std::pair<api::PatchSideRef, api::PatchSideRef>> s1_to_s0 = {
        { { 0, PatchSide::S1 }, { 1, PatchSide::S0 } } };

    Marks orderingMarks()
    {
        std::vector<api::PatchElement> level_0 = block( 0, 3, 5 );
        level_0.push_back( { 1, 3, 3 } );
        return { level_0, { { 0, 8, 8 } } };
    }

    // The renumbered and turned layouts mark the same physical cell as patch_1_alone.
    const Layout patch_1_alone{
        "patch 1 alone fails", { square, right_square }, s1_to_s0, { { { 1, 3, 3 } } }, both( fails( 1, 0, { 3, 3 } ) ), { 0 } };
    const Layout ordering{ "patch 0 fails on level 1, patch 1 on level 0",
                           { square, right_square },
                           s1_to_s0,
                           orderingMarks(),
                           both( fails( 0, 1, { 8, 8 } ) ),
                           { 0, 1 } };
    const Layout renumbered{ "patch 1 alone fails, patches renumbered",
                             { right_square, square },
                             { { { 1, PatchSide::S1 }, { 0, PatchSide::S0 } } },
                             { { { 0, 3, 3 } } },
                             both( fails( 0, 0, { 3, 3 } ) ),
                             { 0 } };
    const Layout turned{ "patch 1 alone fails, patch 1 turned by pi",
                         { square, right_square_turned },
                         { { { 0, PatchSide::S1 }, { 1, PatchSide::S1 } } },
                         { { { 1, 4, 4 } } },
                         both( fails( 1, 0, { 4, 4 } ) ),
                         { 0 } };
    const std::vector<Layout> two_patch = { patch_1_alone, ordering, renumbered, turned };

    std::string describe( const Index& ix )
    {
        std::string out = "(";
        for( size_t i = 0; i < ix.size(); i++ ) out += ( i == 0 ? "" : ", " ) + std::to_string( ix.at( i ) );
        return out + ")";
    }

    std::string describe( const Verdict& v )
    {
        if( not v ) return "satisfied";
        return "fails on patch " + std::to_string( v->patch ) + ", level " + std::to_string( v->level ) + ", cell " +
               describe( v->cell );
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

    // Per patch, per level: the breaks in s and t.
    struct PatchLevels
    {
        std::vector<std::vector<double>> s, t;
    };

    std::vector<PatchLevels> dyadicLevels( const Layout& layout )
    {
        std::vector<PatchLevels> out;
        for( const PatchLayout& patch : layout.patches )
        {
            PatchLevels& p = out.emplace_back();
            for( size_t level = 0; level <= layout.marks.size(); level++ )
            {
                p.s.push_back( levelBreaks( patch.breaks_s, level ) );
                p.t.push_back( levelBreaks( patch.breaks_t, level ) );
            }
        }
        return out;
    }

    // The ( i, j ) cells of each degree q function on one level.
    std::vector<std::set<Index>> levelSupports( const PatchLevels& patch, const size_t level, const size_t q )
    {
        const auto& breaks_s = patch.s.at( level );
        const auto& breaks_t = patch.t.at( level );
        const auto spans_s = supportSpans( openKnots( breaks_s, q ), breaks_s, q );
        const auto spans_t = supportSpans( openKnots( breaks_t, q ), breaks_t, q );
        std::vector<std::set<Index>> out;
        for( const std::vector<size_t>& t_spans : spans_t )
            for( const std::vector<size_t>& s_spans : spans_s )
            {
                std::set<Index> cells;
                for( const size_t i : s_spans )
                    for( const size_t j : t_spans ) cells.insert( { i, j } );
                out.push_back( cells );
            }
        return out;
    }

    std::set<Index> marked( const Marks& marks, const size_t patch, const size_t level )
    {
        std::set<Index> out;
        for( const auto& [p, i, j] : marks.at( level ) )
            if( p == patch ) out.insert( { i, j } );
        return out;
    }

    // The first index fastest, as util::flatten orders cells.
    bool before( const Index& a, const Index& b )
    {
        return std::lexicographical_compare( a.rbegin(), a.rend(), b.rbegin(), b.rend() );
    }

    // P_n from the definition, with R_l the level l marks and supports from the level knots.
    Verdict oracle( const std::vector<PatchLevels>& patches, const Marks& marks, const size_t q )
    {
        for( size_t p = 0; p < patches.size(); p++ )
            for( size_t level = 0; level < marks.size(); level++ )
            {
                const std::set<Index> refined = marked( marks, p, level );
                std::set<Index> covered;
                for( const std::set<Index>& support : levelSupports( patches.at( p ), level, q ) )
                    if( std::includes( refined.begin(), refined.end(), support.begin(), support.end() ) )
                        covered.insert( support.begin(), support.end() );
                std::optional<Index> first;
                for( const Index& c : refined )
                    if( not covered.contains( c ) and ( not first or before( c, *first ) ) ) first = c;
                if( first ) return Failure{ p, level, *first };
            }
        return std::nullopt;
    }

    // (patch, level, i, j) of every leaf: the level 0 cells, each mark replaced by its children.
    std::set<Index> leavesFromMarks( const std::vector<PatchLevels>& patches, const Marks& marks )
    {
        std::set<Index> leaves;
        for( size_t p = 0; p < patches.size(); p++ )
            for( size_t i = 0; i + 1 < patches.at( p ).s.front().size(); i++ )
                for( size_t j = 0; j + 1 < patches.at( p ).t.front().size(); j++ ) leaves.insert( { p, 0, i, j } );
        for( size_t level = 0; level < marks.size(); level++ )
            for( const auto& [p, i, j] : marks.at( level ) )
            {
                const PatchLevels& pl = patches.at( p );
                const size_t r_s = ( pl.s.at( level + 1 ).size() - 1 ) / ( pl.s.at( level ).size() - 1 );
                const size_t r_t = ( pl.t.at( level + 1 ).size() - 1 ) / ( pl.t.at( level ).size() - 1 );
                if( leaves.erase( { p, level, i, j } ) != 1 )
                    throw std::runtime_error( "mark " + describe( { p, level, i, j } ) + " is not a leaf" );
                for( size_t a = 0; a < r_s; a++ )
                    for( size_t b = 0; b < r_t; b++ ) leaves.insert( { p, level + 1, r_s * i + a, r_t * j + b } );
            }
        return leaves;
    }

    // The fixture's premises: the hierarchy's leaves are the marks', so R_l is the level l marks; and on each
    // interior level every oracle support meeting the marks is whole, so no boundary support reaches them.
    std::vector<std::string> premiseProblems( const std::string& prefix,
                                              const std::vector<PatchLevels>& patches,
                                              const Marks& marks,
                                              const std::set<size_t>& interior_levels,
                                              const size_t q,
                                              const std::set<Index>& leaves )
    {
        std::vector<std::string> problems;
        if( leaves != leavesFromMarks( patches, marks ) ) problems.push_back( prefix + "the leaves are not the marks'" );
        for( const size_t level : interior_levels )
        {
            size_t n_marked = 0;
            for( size_t p = 0; p < patches.size(); p++ )
            {
                const std::set<Index> refined = marked( marks, p, level );
                n_marked += refined.size();
                for( const std::set<Index>& support : levelSupports( patches.at( p ), level, q ) )
                {
                    const bool meets = std::any_of(
                        support.begin(), support.end(), [&]( const Index& c ) { return refined.contains( c ); } );
                    if( meets and support.size() != ( q + 1 ) * ( q + 1 ) )
                        problems.push_back( prefix + "a support cut by the boundary meets the level " +
                                            std::to_string( level ) + " marks of patch " + std::to_string( p ) );
                }
            }
            if( n_marked == 0 ) problems.push_back( prefix + "no marks on interior level " + std::to_string( level ) );
        }
        return problems;
    }

    // The oracle against the hand value, then the implementation against the oracle.
    std::vector<std::string> verdictProblems( const std::string& prefix,
                                              const Verdict& expected,
                                              const std::optional<Verdict>& by_hand,
                                              const Verdict& got )
    {
        std::vector<std::string> problems;
        if( by_hand and *by_hand != expected )
            problems.push_back( prefix + "oracle " + describe( expected ) + ", by hand " + describe( *by_hand ) );
        if( got != expected ) problems.push_back( prefix + "got " + describe( got ) + ", expected " + describe( expected ) );
        return problems;
    }

    std::optional<Verdict> byHand( const Layout& layout, const size_t degree )
    {
        const auto it = layout.by_hand.find( degree );
        if( it == layout.by_hand.end() ) return std::nullopt;
        return it->second;
    }

    Verdict verdictOf( const api::PolicyCheck& c, std::vector<std::string>& problems, const std::string& prefix )
    {
        if( c.satisfied() != not c.failure.has_value() ) problems.push_back( prefix + "satisfied() disagrees with failure" );
        if( not c.failure ) return std::nullopt;
        if( c.failure->policy != RefinementPolicy::NFormSupport ) problems.push_back( prefix + "the failure names another policy" );
        return Failure{ c.failure->patch, c.failure->level, Index( c.failure->cell.begin(), c.failure->cell.end() ) };
    }

    // require throws PolicyViolation carrying the check's failure, and returns iff the check is satisfied.
    template <typename F>
    std::vector<std::string> requireProblems( const std::string& prefix, const F& require, const api::PolicyCheck& c )
    {
        try
        {
            require();
            if( c.failure ) return { prefix + "require returned after a failed check" };
        }
        catch( const api::PolicyViolation& v )
        {
            if( not c.failure ) return { prefix + "require refused a satisfied hierarchy" };
            if( v.failure() != *c.failure ) return { prefix + "the violation's failure is not the check's" };
        }
        catch( const std::exception& e )
        {
            return { prefix + "require threw another exception: " + e.what() };
        }
        return {};
    }

    api::PatchGeometry patchGeometry( const PatchLayout& patch, const size_t degree )
    {
        const std::vector<double> knots_s = openKnots( patch.breaks_s, degree );
        const std::vector<double> knots_t = openKnots( patch.breaks_t, degree );
        const auto greville = [&]( const std::vector<double>& knots, const size_t k ) {
            double sum = 0;
            for( size_t a = 1; a <= degree; a++ ) sum += knots.at( k + a );
            return sum / double( degree );
        };
        const size_t n_s = knots_s.size() - degree - 1;
        const size_t n_t = knots_t.size() - degree - 1;
        const AxisMap& m = patch.phi;
        Eigen::MatrixXd cpts( 2, n_s * n_t );
        for( size_t k_t = 0; k_t < n_t; k_t++ )
            for( size_t k_s = 0; k_s < n_s; k_s++ )
            {
                const double s = greville( knots_s, k_s );
                const double t = greville( knots_t, k_t );
                cpts.col( k_s + n_s * k_t ) << m.x0 + m.xs * s + m.xt * t, m.y0 + m.ys * s + m.yt * t;
            }
        return { KnotVector( knots_s, 1e-10 ), KnotVector( knots_t, 1e-10 ), degree, degree, cpts };
    }

    std::unique_ptr<api::NavierStokesHierarchicalMultiPatchDiscretization> build( const Layout& layout,
                                                                                 const size_t degree )
    {
        std::vector<api::PatchGeometry> patches;
        for( const PatchLayout& patch : layout.patches ) patches.push_back( patchGeometry( patch, degree ) );
        return std::make_unique<api::NavierStokesHierarchicalMultiPatchDiscretization>( patches, layout.gluing, layout.marks );
    }

    std::unique_ptr<api::NavierStokesHierarchicalDiscretization> buildSingle( const Layout& layout, const size_t degree )
    {
        if( layout.patches.size() != 1 ) throw std::runtime_error( "not a one patch layout" );
        const api::PatchGeometry g = patchGeometry( layout.patches.front(), degree );
        std::vector<std::vector<std::pair<size_t, size_t>>> marks;
        for( const auto& level : layout.marks )
        {
            marks.emplace_back();
            for( const auto& [p, i, j] : level ) marks.back().emplace_back( i, j );
        }
        return std::make_unique<api::NavierStokesHierarchicalDiscretization>(
            g.kv_s, g.kv_t, degree, degree, g.control_points, marks );
    }

    std::set<Index> leaves( const HierarchicalMultiPatchCombinatorialMap& cmap )
    {
        std::set<Index> out;
        for( const auto& info : leafElementInfos( cmap ) )
            out.insert( { info.patch, info.level, info.indices.at( 0 ), info.indices.at( 1 ) } );
        return out;
    }

    std::set<Index> leaves( const HierarchicalTPCombinatorialMap& cmap )
    {
        std::set<Index> out;
        for( const auto& info : leafElementInfos( cmap ) )
            out.insert( { 0, info.level, info.indices.at( 0 ), info.indices.at( 1 ) } );
        return out;
    }

    template <typename Check>
    void checkEachFixture( const std::vector<Layout>& layouts, const Check& check )
    {
        for( const Layout& layout : layouts )
            for( const size_t degree : fixture_degrees )
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

    // The message of an E thrown by f, or a description of what happened instead.
    template <typename E, typename F>
    std::string refusal( const F& f )
    {
        try
        {
            f();
        }
        catch( const E& e )
        {
            return e.what();
        }
        catch( const std::exception& e )
        {
            return std::string( "another exception: " ) + e.what();
        }
        return "no exception";
    }

    std::vector<std::string> multiPatchProblems( const std::string& prefix, const Layout& layout, const size_t degree )
    {
        const auto disc = build( layout, degree );
        const std::vector<PatchLevels> levels = dyadicLevels( layout );
        std::vector<std::string> problems = premiseProblems(
            prefix, levels, layout.marks, layout.interior_levels, degree - 1, leaves( disc->L2_ss.basisComplex().parametricAtlas().cmap() ) );
        const api::PolicyCheck c = api::checkRefinementPolicy( *disc, RefinementPolicy::NFormSupport );
        const Verdict got = verdictOf( c, problems, prefix );
        const auto v = verdictProblems( prefix, oracle( levels, layout.marks, degree - 1 ), byHand( layout, degree ), got );
        problems.insert( problems.end(), v.begin(), v.end() );
        const auto r = requireProblems(
            prefix, [&] { api::requireRefinementPolicy( *disc, RefinementPolicy::NFormSupport ); }, c );
        problems.insert( problems.end(), r.begin(), r.end() );
        return problems;
    }

    TPSplineSpace bspline( const std::vector<double>& breaks, const size_t q, const size_t dim )
    {
        const KnotVector kv( openKnots( breaks, q ), 1e-10 );
        SmallVector<KnotVector, 3> kvs;
        SmallVector<size_t, 3> degrees;
        for( size_t k = 0; k < dim; k++ )
        {
            kvs.push_back( kv );
            degrees.push_back( q );
        }
        return buildBSpline( kvs, degrees );
    }

    Cell levelCell( const TPSplineSpace& level, const Index& ix )
    {
        SmallVector<Dart, 3> darts;
        for( const size_t i : ix ) darts.push_back( Dart( i ) );
        const TPCombinatorialMap& cmap = level.basisComplex().parametricAtlas().cmap();
        return Cell( flattenFull( cmap, FullyUnflattenedDart( darts ) ), cmap.dim() );
    }
}

TEST_CASE( "P_n on single patch hierarchies: check and require against the oracle", "[nform_policy]" )
{
    checkEachFixture( one_patch, []( const std::string& prefix, const Layout& layout, const size_t degree ) {
        const auto disc = buildSingle( layout, degree );
        const std::vector<PatchLevels> levels = dyadicLevels( layout );
        std::vector<std::string> problems = premiseProblems(
            prefix, levels, layout.marks, layout.interior_levels, degree - 1, leaves( disc->L2_ss.basisComplex().parametricAtlas().cmap() ) );
        const api::PolicyCheck c = api::checkRefinementPolicy( *disc, RefinementPolicy::NFormSupport );
        const Verdict got = verdictOf( c, problems, prefix );
        const auto v = verdictProblems( prefix, oracle( levels, layout.marks, degree - 1 ), byHand( layout, degree ), got );
        problems.insert( problems.end(), v.begin(), v.end() );
        const auto r = requireProblems(
            prefix, [&] { api::requireRefinementPolicy( *disc, RefinementPolicy::NFormSupport ); }, c );
        problems.insert( problems.end(), r.begin(), r.end() );
        return problems;
    } );
}

TEST_CASE( "P_n on multipatch hierarchies: check and require against the oracle", "[nform_policy]" )
{
    checkEachFixture( { side_by_side, stacked, half_turn, quarter_turn }, multiPatchProblems );
    checkEachFixture( one_patch, multiPatchProblems );
    checkEachFixture( two_patch, multiPatchProblems );
}

TEST_CASE( "P_n failures stay in patch-local numbering under renumbering and a half turn", "[nform_policy]" )
{
    // The three layouts mark the same physical cell, [11, 12] x [3, 4].
    checkEachFixture( { patch_1_alone, renumbered, turned },
                      []( const std::string& prefix, const Layout& layout, const size_t degree ) {
                          const auto disc = build( layout, degree );
                          const api::PolicyCheck c = api::checkRefinementPolicy( *disc, RefinementPolicy::NFormSupport );
                          if( not c.failure ) return std::vector<std::string>{ prefix + "satisfied" };
                          const AxisMap& m = layout.patches.at( c.failure->patch ).phi;
                          const double s = c.failure->cell.at( 0 ) + 0.5;
                          const double t = c.failure->cell.at( 1 ) + 0.5;
                          const double x = m.x0 + m.xs * s + m.xt * t;
                          const double y = m.y0 + m.ys * s + m.yt * t;
                          if( std::abs( x - 11.5 ) > 1e-12 or std::abs( y - 3.5 ) > 1e-12 )
                              return std::vector<std::string>{ prefix + "the failing cell is not the marked one" };
                          return std::vector<std::string>{};
                      } );
}

TEST_CASE( "P_n follows the parent relation at a refinement ratio of 3", "[nform_policy]" )
{
    // q = 1 on [0, 2]^2; levels of 2, 6 (ratio 3) and 12 (ratio 2) cells a side.  Dyadic children of the level
    // 0 mark would miss the level 1 mark.
    const size_t q = 1;
    const Marks marks = { { { 0, 0, 0 } }, { { 0, 2, 2 } } };
    const Verdict by_hand = fails( 0, 1, { 2, 2 } );
    std::vector<std::string> problems;
    try
    {
        PatchLevels pl;
        std::vector<std::shared_ptr<const TPSplineSpace>> levels;
        for( const size_t n : { 2, 6, 12 } )
        {
            std::vector<double> breaks;
            for( size_t i = 0; i <= n; i++ ) breaks.push_back( 2.0 * double( i ) / double( n ) );
            pl.s.push_back( breaks );
            pl.t.push_back( breaks );
            levels.push_back( std::make_shared<const TPSplineSpace>( bspline( breaks, q, 2 ) ) );
        }
        const std::set<Index> expected_leaves = leavesFromMarks( { pl }, marks );
        std::vector<std::vector<Cell>> leaf_cells( levels.size() );
        for( const Index& k : expected_leaves )
            leaf_cells.at( k.at( 1 ) ).push_back( levelCell( *levels.at( k.at( 1 ) ), { k.at( 2 ), k.at( 3 ) } ) );
        const HierarchicalTPSplineSpace ss = buildHierarchicalSplineSpace( levels, leaf_cells );

        const HierarchicalTPCombinatorialMap& cmap = ss.basisComplex().parametricAtlas().cmap();
        if( cmap.refinementRatios() != std::vector<SmallVector<size_t, 3>>{ { 3, 3 }, { 2, 2 } } )
            problems.push_back( "the refinement ratios are not 3 and 2" );
        const auto p = premiseProblems( "", { pl }, marks, { 1 }, q, leaves( cmap ) );
        problems.insert( problems.end(), p.begin(), p.end() );

        const auto f = checkNFormSupportPolicy( ss );
        const Verdict got = f ? Verdict( Failure{ f->patch, f->level, Index( f->cell.begin(), f->cell.end() ) } ) : std::nullopt;
        const auto v = verdictProblems( "", oracle( { pl }, marks, q ), by_hand, got );
        problems.insert( problems.end(), v.begin(), v.end() );
    }
    catch( const std::exception& e )
    {
        problems.push_back( std::string( "throws: " ) + e.what() );
    }
    INFO( join( problems ) );
    CHECK( problems.empty() );
}

TEST_CASE( "PolicyViolation names the failure", "[nform_policy]" )
{
    const auto disc = buildSingle( three_levels, 2 );
    CHECK( refusal<api::PolicyViolation>( [&] { api::requireRefinementPolicy( *disc, RefinementPolicy::NFormSupport ); } ) ==
           "Refinement policy NFormSupport is violated on patch 0, level 1, cell (8, 8)." );
}

TEST_CASE( "Unsupported refinement policies raise PolicyNotImplemented before any check", "[nform_policy]" )
{
    // A satisfied and a failing hierarchy of each kind; NFormSupport tells them apart.
    const auto single_ok = buildSingle( corner, 2 );
    const auto single_bad = buildSingle( interior, 2 );
    const auto multi_ok = build( corner, 2 );
    const auto multi_bad = build( patch_1_alone, 2 );
    CHECK( api::checkRefinementPolicy( *single_ok, RefinementPolicy::NFormSupport ).satisfied() );
    CHECK_FALSE( api::checkRefinementPolicy( *single_bad, RefinementPolicy::NFormSupport ).satisfied() );
    CHECK( api::checkRefinementPolicy( *multi_ok, RefinementPolicy::NFormSupport ).satisfied() );
    CHECK_FALSE( api::checkRefinementPolicy( *multi_bad, RefinementPolicy::NFormSupport ).satisfied() );

    for( const RefinementPolicy policy : { RefinementPolicy::ZeroFormSupport, RefinementPolicy::EM, RefinementPolicy::MHD } )
    {
        const std::string expected = "Refinement policy " + api::policyName( policy ) + " is not implemented.";
        CAPTURE( expected );
        for( const auto* disc : { single_ok.get(), single_bad.get() } )
        {
            CHECK( refusal<api::PolicyNotImplemented>( [&] { api::checkRefinementPolicy( *disc, policy ); } ) == expected );
            CHECK( refusal<api::PolicyNotImplemented>( [&] { api::requireRefinementPolicy( *disc, policy ); } ) == expected );
        }
        for( const auto* disc : { multi_ok.get(), multi_bad.get() } )
        {
            CHECK( refusal<api::PolicyNotImplemented>( [&] { api::checkRefinementPolicy( *disc, policy ); } ) == expected );
            CHECK( refusal<api::PolicyNotImplemented>( [&] { api::requireRefinementPolicy( *disc, policy ); } ) == expected );
        }
    }
}

TEST_CASE( "A value outside RefinementPolicy is malformed input", "[nform_policy]" )
{
    const auto single = buildSingle( interior, 2 );
    const auto multi = build( patch_1_alone, 2 );
    const RefinementPolicy unknown = static_cast<RefinementPolicy>( 4 );
    const std::string expected = "Unknown refinement policy.";
    CHECK( refusal<std::invalid_argument>( [&] { api::policyName( unknown ); } ) == expected );
    CHECK( refusal<std::invalid_argument>( [&] { api::checkRefinementPolicy( *single, unknown ); } ) == expected );
    CHECK( refusal<std::invalid_argument>( [&] { api::requireRefinementPolicy( *single, unknown ); } ) == expected );
    CHECK( refusal<std::invalid_argument>( [&] { api::checkRefinementPolicy( *multi, unknown ); } ) == expected );
    CHECK( refusal<std::invalid_argument>( [&] { api::requireRefinementPolicy( *multi, unknown ); } ) == expected );
}

TEST_CASE( "P_n is not implemented for a 3d space", "[nform_policy]" )
{
    // A valid two level 3d hierarchy, one corner cell refined; the 2d one built the same way is checked.
    for( const size_t dim : { 2, 3 } )
    {
        CAPTURE( dim );
        std::vector<std::shared_ptr<const TPSplineSpace>> levels;
        for( const std::vector<double>& breaks : { std::vector<double>{ 0, 1, 2 }, std::vector<double>{ 0, 0.5, 1, 1.5, 2 } } )
            levels.push_back( std::make_shared<const TPSplineSpace>( bspline( breaks, 1, dim ) ) );
        std::vector<std::vector<Cell>> leaf_cells( 2 );
        util::iterateTensorProduct( util::IndexVec( dim, 2 ), [&]( const util::IndexVec& iv ) {
            const Index ix( iv.begin(), iv.end() );
            if( std::any_of( ix.begin(), ix.end(), []( const size_t i ) { return i != 0; } ) )
                leaf_cells.at( 0 ).push_back( levelCell( *levels.at( 0 ), ix ) );
            else
                util::iterateTensorProduct( util::IndexVec( dim, 2 ), [&]( const util::IndexVec& c ) {
                    leaf_cells.at( 1 ).push_back( levelCell( *levels.at( 1 ), Index( c.begin(), c.end() ) ) );
                } );
        } );
        const HierarchicalTPSplineSpace ss = buildHierarchicalSplineSpace( levels, leaf_cells );
        const std::string got = refusal<NFormSupportNotImplemented>( [&] { checkNFormSupportPolicy( ss ); } );
        CHECK( got == ( dim == 2 ? "no exception" : "The n-form support policy is implemented only for 2d spaces." ) );
    }
}
