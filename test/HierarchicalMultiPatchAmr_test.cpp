// Item 1, step 10.5 (outstanding_issues.txt): the policy-free hierarchical multipatch discretization and the
// per-patch leaf, support and activity queries an adaptive loop reads, each against an oracle built here from
// the marks, the knots or the patch maps.
#include <catch2/catch_test_macros.hpp>
#include <NavierStokesDiscretization.hpp>
#include <HierarchicalMultiPatchCombinatorialMap.hpp>
#include <HierarchicalMultiPatchSplineSpace.hpp>
#include <CombinatorialMapMethods.hpp>
#include <KnotVector.hpp>
#include <ParentPoint.hpp>
#include <TPSplineSpace.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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
    using api::PatchSide;

    // x = x0 + xs s + xt t, y = y0 + ys s + yt t, axis aligned.
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

    struct Layout
    {
        std::string name;
        std::vector<PatchLayout> patches;
        std::vector<std::pair<api::PatchSideRef, api::PatchSideRef>> gluing;
        std::vector<std::vector<api::PatchElement>> marks;
    };

    const AxisMap identity{ 0, 1, 0, 0, 0, 1 };

    // Patch 0 refined at the interface, jump 2 against patch 1's level 0 leaf.
    const Layout side_by_side{ "side by side, S1 to S0",
                               { { { 0, 1, 2 }, { 0, 1, 3 }, identity },
                                 { { 0, 1, 2, 3 }, { 0, 1, 3 }, { 2, 1, 0, 0, 0, 1 } } },
                               { { { 0, PatchSide::S1 }, { 1, PatchSide::S0 } } },
                               { { { 0, 1, 0 } }, { { 0, 3, 0 }, { 0, 3, 1 } } } };

    // Patch 1's s is translated; patch 1 refined at the interface, jump 2.
    const Layout stacked{ "stacked, T1 to T0, translated s",
                          { { { 0, 1, 2 }, { 0, 1 }, identity }, { { 5, 6, 7 }, { 0, 2 }, { -5, 1, 0, 1, 0, 1 } } },
                          { { { 0, PatchSide::T1 }, { 1, PatchSide::T0 } } },
                          { { { 1, 1, 0 } }, { { 1, 2, 0 } } } };

    // Patch 1 turned by pi, its t pattern reversed; both sides refined at different parts of the interface.  Patch
    // 0's (0, 0) mark keeps the two patches' local active ids apart at p = 3.
    const Layout half_turn{ "half turn, S1 to S1",
                            { { { 0, 1, 2 }, { 0, 1, 3 }, identity }, { { 0, 1, 2 }, { 0, 2, 3 }, { 4, -1, 0, 3, 0, -1 } } },
                            { { { 0, PatchSide::S1 }, { 1, PatchSide::S1 } } },
                            { { { 0, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 } }, { { 1, 3, 1 } } } };

    // Patch 1 turned by pi / 2, so patch 0's t meets patch 1's s; both sides refined.
    const Layout quarter_turn{ "quarter turn, S1 to T0",
                               { { { 0, 1, 2 }, { 0, 1, 3 }, identity }, { { 0, 2, 3 }, { 0, 1, 2 }, { 2, 0, 1, 3, -1, 0 } } },
                               { { { 0, PatchSide::S1 }, { 1, PatchSide::T0 } } },
                               { { { 1, 1, 0 }, { 0, 1, 1 } } } };

    const std::vector<Layout> layouts = { side_by_side, stacked, half_turn, quarter_turn };
    const std::vector<size_t> fixture_degrees = { 2, 3 };

    std::string describe( const Index& ix )
    {
        std::string out = "(";
        for( size_t i = 0; i < ix.size(); i++ ) out += ( i == 0 ? "" : ", " ) + std::to_string( ix.at( i ) );
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

    // Sorted ( i, j ) cell supports of the degree q functions of one patch level, function id k_s + n_s k_t.
    std::vector<std::vector<Index>> levelSupports( const PatchLayout& patch, const size_t level, const size_t q )
    {
        const std::vector<double> breaks_s = levelBreaks( patch.breaks_s, level );
        const std::vector<double> breaks_t = levelBreaks( patch.breaks_t, level );
        const auto spans_s = supportSpans( openKnots( breaks_s, q ), breaks_s, q );
        const auto spans_t = supportSpans( openKnots( breaks_t, q ), breaks_t, q );
        std::vector<std::vector<Index>> out;
        for( const std::vector<size_t>& t_spans : spans_t )
            for( const std::vector<size_t>& s_spans : spans_s )
            {
                std::vector<Index> cells;
                for( const size_t i : s_spans )
                    for( const size_t j : t_spans ) cells.push_back( { i, j } );
                std::sort( cells.begin(), cells.end() );
                out.push_back( cells );
            }
        return out;
    }

    // (level, i, j) of every leaf of one patch: all level 0 cells, each mark replaced by its children.
    std::set<Index> patchLeavesFromMarks( const PatchLayout& patch,
                                          const size_t patch_id,
                                          const std::vector<std::vector<api::PatchElement>>& marks )
    {
        std::set<Index> leaves;
        for( size_t i = 0; i + 1 < patch.breaks_s.size(); i++ )
            for( size_t j = 0; j + 1 < patch.breaks_t.size(); j++ ) leaves.insert( { 0, i, j } );
        for( size_t level = 0; level < marks.size(); level++ )
            for( const auto& [p, i, j] : marks.at( level ) )
            {
                if( p != patch_id ) continue;
                if( leaves.erase( { level, i, j } ) != 1 )
                    throw std::runtime_error( "mark " + describe( { p, level, i, j } ) + " is not a leaf" );
                for( const size_t ci : { 2 * i, 2 * i + 1 } )
                    for( const size_t cj : { 2 * j, 2 * j + 1 } ) leaves.insert( { level + 1, ci, cj } );
            }
        return leaves;
    }

    // (patch, level, i, j) of every leaf.
    std::set<Index> leavesFromMarks( const Layout& layout )
    {
        std::set<Index> out;
        for( size_t p = 0; p < layout.patches.size(); p++ )
            for( const Index& k : patchLeavesFromMarks( layout.patches.at( p ), p, layout.marks ) )
                out.insert( { p, k.at( 0 ), k.at( 1 ), k.at( 2 ) } );
        return out;
    }

    // Coefficients of c + c_s s + c_ss s^2 + c_t t + c_tt t^2.
    struct Quadratic
    {
        double c, c_s, c_ss, c_t, c_tt;
    };

    constexpr double c_x = 0.1;
    constexpr double c_y = 0.05;

    // F( x, y ) = ( x + c_x y^2, y + c_y x^2 ).
    Eigen::Vector2d mapped( const AxisMap& m, const double s, const double t )
    {
        const double x = m.x0 + m.xs * s + m.xt * t;
        const double y = m.y0 + m.ys * s + m.yt * t;
        return { x + c_x * y * y, y + c_y * x * x };
    }

    // F after an axis aligned map has no s t term.
    std::array<Quadratic, 2> mappedInST( const AxisMap& m )
    {
        if( m.xs * m.xt != 0 or m.ys * m.yt != 0 ) throw std::logic_error( "patch map is not axis aligned" );
        const auto square = []( const double a, const double b_s, const double b_t ) {
            return Quadratic{ a * a, 2 * a * b_s, b_s * b_s, 2 * a * b_t, b_t * b_t };
        };
        const Quadratic x2 = square( m.x0, m.xs, m.xt );
        const Quadratic y2 = square( m.y0, m.ys, m.yt );
        return { Quadratic{ m.x0 + c_x * y2.c, m.xs + c_x * y2.c_s, c_x * y2.c_ss, m.xt + c_x * y2.c_t, c_x * y2.c_tt },
                 Quadratic{ m.y0 + c_y * x2.c, m.ys + c_y * x2.c_s, c_y * x2.c_ss, m.yt + c_y * x2.c_t, c_y * x2.c_tt } };
    }

    // Blossoms of u and u^2 at the p inner knots of function k: its coefficients of u and u^2, for p >= 2.
    std::pair<double, double> blossoms( const std::vector<double>& knots, const size_t p, const size_t k )
    {
        double sum = 0;
        double pairs = 0;
        for( size_t a = k + 1; a <= k + p; a++ )
        {
            sum += knots.at( a );
            for( size_t b = a + 1; b <= k + p; b++ ) pairs += knots.at( a ) * knots.at( b );
        }
        return { sum / p, pairs / ( 0.5 * p * ( p - 1 ) ) };
    }

    // The unrefined control points of F after the patch map, exact for the quadratic field.
    api::PatchGeometry patchGeometry( const PatchLayout& patch, const size_t degree )
    {
        const std::vector<double> knots_s = openKnots( patch.breaks_s, degree );
        const std::vector<double> knots_t = openKnots( patch.breaks_t, degree );
        const size_t n_s = knots_s.size() - degree - 1;
        const size_t n_t = knots_t.size() - degree - 1;
        const std::array<Quadratic, 2> f = mappedInST( patch.phi );
        Eigen::MatrixXd cpts( 2, n_s * n_t );
        for( size_t k_t = 0; k_t < n_t; k_t++ )
            for( size_t k_s = 0; k_s < n_s; k_s++ )
            {
                const auto [s1, s2] = blossoms( knots_s, degree, k_s );
                const auto [t1, t2] = blossoms( knots_t, degree, k_t );
                for( size_t r = 0; r < 2; r++ )
                    cpts( r, k_s + n_s * k_t ) =
                        f.at( r ).c + f.at( r ).c_s * s1 + f.at( r ).c_ss * s2 + f.at( r ).c_t * t1 + f.at( r ).c_tt * t2;
            }
        return { KnotVector( knots_s, 1e-10 ), KnotVector( knots_t, 1e-10 ), degree, degree, cpts };
    }

    // The patch map itself, exact at the Greville points.
    api::PatchGeometry affineGeometry( const PatchLayout& patch, const size_t degree )
    {
        api::PatchGeometry g = patchGeometry( patch, degree );
        const std::vector<double> knots_s = openKnots( patch.breaks_s, degree );
        const std::vector<double> knots_t = openKnots( patch.breaks_t, degree );
        const size_t n_s = knots_s.size() - degree - 1;
        const AxisMap& m = patch.phi;
        for( Eigen::Index k = 0; k < g.control_points.cols(); k++ )
        {
            const double s = blossoms( knots_s, degree, k % n_s ).first;
            const double t = blossoms( knots_t, degree, k / n_s ).first;
            g.control_points.col( k ) << m.x0 + m.xs * s + m.xt * t, m.y0 + m.ys * s + m.yt * t;
        }
        return g;
    }

    std::vector<api::PatchGeometry> patchGeometries( const Layout& layout, const size_t degree )
    {
        std::vector<api::PatchGeometry> out;
        for( const PatchLayout& patch : layout.patches ) out.push_back( patchGeometry( patch, degree ) );
        return out;
    }

    std::unique_ptr<api::NavierStokesHierarchicalMultiPatchDiscretization> build( const Layout& layout,
                                                                                 const size_t degree )
    {
        return std::make_unique<api::NavierStokesHierarchicalMultiPatchDiscretization>(
            patchGeometries( layout, degree ), layout.gluing, layout.marks );
    }

    Index keyOf( const MultiPatchLeafElementInfo& info )
    {
        Index key = { info.patch, info.level };
        key.insert( key.end(), info.indices.begin(), info.indices.end() );
        return key;
    }

    // leafElementInfos against the marks, and leafElementInfo from every dart of every leaf.
    std::vector<std::string> leafInfoProblems( const std::string& prefix,
                                               const HierarchicalMultiPatchCombinatorialMap& cmap,
                                               const std::set<Index>& expected )
    {
        std::vector<std::string> problems;
        std::multiset<Index> actual;
        std::vector<Cell> handles;
        iterateCellsWhile( cmap, cmap.dim(), [&]( const Cell& c ) {
            handles.push_back( c );
            return true;
        } );
        const std::vector<MultiPatchLeafElementInfo> infos = leafElementInfos( cmap );
        if( infos.size() != handles.size() )
            problems.push_back( prefix + std::to_string( infos.size() ) + " infos of " +
                                std::to_string( handles.size() ) + " leaves" );

        size_t other_level_interface_darts = 0;
        for( size_t n = 0; n < infos.size(); n++ )
        {
            const MultiPatchLeafElementInfo& info = infos.at( n );
            const Index key = keyOf( info );
            actual.insert( key );
            if( n < handles.size() and not( info.leaf == handles.at( n ) ) )
                problems.push_back( prefix + describe( key ) + " is out of iterateCellsWhile order" );
            if( info.indices.size() != cmap.dim() ) problems.push_back( prefix + describe( key ) + " has the wrong length" );
            iterateDartsOfCell( cmap, info.leaf, [&]( const Dart& d ) {
                const auto [patch, constituent_d] = cmap.dartRanges().toLocalDart( d );
                const size_t dart_level = cmap.constituents().at( patch )->dartRanges().toLocalDart( constituent_d ).first;
                const std::optional<Dart> across = cmap.phi( cmap.dim(), d );
                if( dart_level != info.level and across and cmap.dartRanges().toLocalDart( *across ).first != patch )
                    other_level_interface_darts++;

                const MultiPatchLeafElementInfo by_dart = leafElementInfo( cmap, Cell( d, cmap.dim() ) );
                if( keyOf( by_dart ) != key )
                    problems.push_back( prefix + "dart " + std::to_string( d.id() ) + " of " + describe( key ) +
                                        " gives " + describe( keyOf( by_dart ) ) );
                if( by_dart.leaf.dart() != d )
                    problems.push_back( prefix + "dart " + std::to_string( d.id() ) + " handle not preserved" );
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
        if( other_level_interface_darts == 0 )
            problems.push_back( prefix + "no interface dart of a leaf lies on another level" );
        return problems;
    }

    std::vector<std::string> supportProblems( const std::string& prefix,
                                              const std::vector<std::vector<SmallVector<size_t, 3>>>& actual,
                                              const std::vector<std::vector<Index>>& expected )
    {
        if( actual.size() != expected.size() )
            return { prefix + std::to_string( actual.size() ) + " supports; the knot oracle has " +
                     std::to_string( expected.size() ) };

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

    // Per patch active functions against: the multipatch level function's support, the union of its patch
    // supports, in Omega^l and not in Omega^(l+1).  Glued level functions are read from the level space.
    std::vector<std::string> activeProblems( const std::string& prefix,
                                             const HierarchicalMultiPatchSplineSpace& ss,
                                             const Layout& layout,
                                             const size_t q,
                                             const bool glued )
    {
        const size_t n_levels = layout.marks.size() + 1;
        if( ss.refinementLevels().size() != n_levels )
            return { prefix + std::to_string( ss.refinementLevels().size() ) + " levels" };

        std::vector<std::set<Index>> marked( n_levels );
        for( size_t level = 0; level < layout.marks.size(); level++ )
            for( const auto& [p, i, j] : layout.marks.at( level ) ) marked.at( level ).insert( { p, i, j } );

        std::vector<std::string> problems;
        for( size_t level = 0; level < n_levels; level++ )
        {
            const std::vector<std::vector<FunctionId>>& fmap = ss.refinementLevels().at( level )->functionIdMap();
            std::map<size_t, std::vector<Index>> supports;
            for( size_t p = 0; p < layout.patches.size(); p++ )
            {
                const std::vector<std::vector<Index>> patch_supports = levelSupports( layout.patches.at( p ), level, q );
                if( patch_supports.size() != fmap.at( p ).size() )
                    return { prefix + "level " + std::to_string( level ) + " patch " + std::to_string( p ) +
                             " has a function map of the wrong size" };
                for( size_t k = 0; k < patch_supports.size(); k++ )
                    for( const Index& cell : patch_supports.at( k ) )
                        supports[fmap.at( p ).at( k ).id()].push_back( { p, cell.at( 0 ), cell.at( 1 ) } );
            }

            const auto in_omega = [&]( const Index& c ) {
                return level == 0 or marked.at( level - 1 ).contains( { c.at( 0 ), c.at( 1 ) / 2, c.at( 2 ) / 2 } );
            };
            const auto in_finer_omega = [&]( const Index& c ) { return marked.at( level ).contains( c ); };

            std::set<size_t> active;
            size_t spanning = 0;
            for( const auto& [g, cells] : supports )
            {
                if( std::any_of( cells.begin(), cells.end(), [&]( const Index& c ) { return c.at( 0 ) != cells.front().at( 0 ); } ) )
                    spanning++;
                if( std::all_of( cells.begin(), cells.end(), in_omega ) and
                    not std::all_of( cells.begin(), cells.end(), in_finer_omega ) )
                    active.insert( g );
            }
            if( glued and spanning == 0 )
                problems.push_back( prefix + "level " + std::to_string( level ) + ": no function spans two patches" );
            if( not glued and spanning != 0 )
                problems.push_back( prefix + "level " + std::to_string( level ) + ": " + std::to_string( spanning ) +
                                    " functions span two patches" );

            for( size_t p = 0; p < layout.patches.size(); p++ )
            {
                std::vector<size_t> expected;
                for( size_t k = 0; k < fmap.at( p ).size(); k++ )
                    if( active.contains( fmap.at( p ).at( k ).id() ) ) expected.push_back( k );

                std::vector<size_t> actual;
                for( const FunctionId& fid : patchLevelActiveFunctions( ss, p, level ) ) actual.push_back( fid.id() );

                if( actual != expected )
                {
                    std::string diff;
                    if( not std::is_sorted( actual.begin(), actual.end() ) ) diff += " not ascending;";
                    for( const size_t f : actual )
                        if( not std::binary_search( expected.begin(), expected.end(), f ) ) diff += " extra " + std::to_string( f );
                    for( const size_t f : expected )
                        if( std::find( actual.begin(), actual.end(), f ) == actual.end() ) diff += " missing " + std::to_string( f );
                    problems.push_back( prefix + "patch " + std::to_string( p ) + " level " + std::to_string( level ) +
                                        " active functions differ:" + diff );
                }
            }
        }
        return problems;
    }

    // The geometry at the center and quarter points of every leaf against F after the patch map.  The quarter
    // points are a set every symmetry of the square maps to itself, so the match is blind to the leaf's dart.
    std::vector<std::string> geometryProblems( const std::string& prefix,
                                               api::NavierStokesHierarchicalMultiPatchDiscretization& nsd,
                                               const Layout& layout )
    {
        const auto& atlas = nsd.H1_ss.basisComplex().parametricAtlas();
        eval::SplineSpaceEvaluator& evaler = nsd.getGeometry();
        const auto close = []( const Eigen::Vector2d& a, const Eigen::Vector2d& b ) {
            return ( a - b ).norm() <= 1e-10 * std::max( 1.0, b.norm() );
        };

        std::vector<std::string> problems;
        size_t leaves = 0;
        for( const MultiPatchLeafElementInfo& info : leafElementInfos( atlas.cmap() ) )
        {
            const PatchLayout& patch = layout.patches.at( info.patch );
            const std::vector<double> bs = levelBreaks( patch.breaks_s, info.level );
            const std::vector<double> bt = levelBreaks( patch.breaks_t, info.level );
            const size_t i = info.indices.at( 0 );
            const size_t j = info.indices.at( 1 );
            const auto expected_at = [&]( const double fs, const double ft ) {
                return mapped( patch.phi,
                               bs.at( i ) + fs * ( bs.at( i + 1 ) - bs.at( i ) ),
                               bt.at( j ) + ft * ( bt.at( j + 1 ) - bt.at( j ) ) );
            };

            evaler.localizeElement( info.leaf );
            const auto evaluated_at = [&]( const double u, const double v ) -> Eigen::Vector2d {
                evaler.localizeParentPoint( param::ParentPoint(
                    atlas.parentDomain( info.leaf ), Eigen::Vector2d( u, v ), { false, false, false, false } ) );
                return evaler.evaluateManifold( nsd.geometryControlPoints() );
            };

            if( not close( evaluated_at( 0.5, 0.5 ), expected_at( 0.5, 0.5 ) ) )
                problems.push_back( prefix + describe( keyOf( info ) ) + " center is off the patch map" );

            std::vector<Eigen::Vector2d> expected;
            for( const double fs : { 0.25, 0.75 } )
                for( const double ft : { 0.25, 0.75 } ) expected.push_back( expected_at( fs, ft ) );
            std::vector<bool> matched( expected.size(), false );
            for( const double u : { 0.25, 0.75 } )
                for( const double v : { 0.25, 0.75 } )
                {
                    const Eigen::Vector2d pt = evaluated_at( u, v );
                    bool found = false;
                    for( size_t k = 0; k < expected.size() and not found; k++ )
                        if( not matched.at( k ) and close( pt, expected.at( k ) ) ) matched.at( k ) = found = true;
                    if( not found )
                        problems.push_back( prefix + describe( keyOf( info ) ) + " parent (" + std::to_string( u ) +
                                            ", " + std::to_string( v ) + ") is off the patch map" );
                }
            leaves++;
        }
        if( leaves == 0 ) problems.push_back( prefix + "no leaves" );
        return problems;
    }

    template <typename Check>
    void checkEachFixture( const Check& check )
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

    bool contains( const std::string& s, const std::string& part ) { return s.find( part ) != std::string::npos; }
}

TEST_CASE( "Multipatch leaf element info: patch and level-local indices from any dart, against the marks",
           "[amr_queries][multipatch]" )
{
    checkEachFixture( []( const std::string& prefix, const Layout& layout, const size_t degree ) {
        const auto nsd = build( layout, degree );
        return leafInfoProblems( prefix, nsd->H1_ss.basisComplex().parametricAtlas().cmap(), leavesFromMarks( layout ) );
    } );
}

TEST_CASE( "Multipatch leaf element info refuses a cell that is not top-dimensional", "[amr_queries][multipatch]" )
{
    const auto nsd = build( side_by_side, 2 );
    const HierarchicalMultiPatchCombinatorialMap& cmap = nsd->H1_ss.basisComplex().parametricAtlas().cmap();
    const Cell leaf = leafElementInfos( cmap ).back().leaf;
    CHECK_NOTHROW( leafElementInfo( cmap, leaf ) );
    for( const uint dim : { 0u, 1u, 3u } )
    {
        CAPTURE( dim );
        CHECK_THROWS_AS( leafElementInfo( cmap, Cell( leaf.dart(), dim ) ), std::invalid_argument );
    }
}

TEST_CASE( "The multipatch L2 hierarchy has the H1 dart structure", "[amr_queries][multipatch]" )
{
    checkEachFixture( []( const std::string& prefix, const Layout& layout, const size_t degree ) {
        const auto nsd = build( layout, degree );
        const HierarchicalMultiPatchCombinatorialMap& h1 = nsd->H1_ss.basisComplex().parametricAtlas().cmap();
        const HierarchicalMultiPatchCombinatorialMap& l2 = nsd->L2_ss.basisComplex().parametricAtlas().cmap();
        std::vector<std::string> problems;
        if( h1.maxDartId() != l2.maxDartId() ) problems.push_back( prefix + "max dart ids differ" );

        const auto dartsOf = []( const CombinatorialMap& cmap ) {
            std::vector<Dart> out;
            cmap.iterateDartsWhile( [&]( const Dart& d ) {
                out.push_back( d );
                return true;
            } );
            return out;
        };
        const std::vector<Dart> darts = dartsOf( h1 );
        if( darts != dartsOf( l2 ) ) problems.push_back( prefix + "dart sets differ" );

        size_t phi_differences = 0;
        for( const Dart& d : darts )
            for( const int i : { -1, 1, 2 } )
                if( h1.phi( i, d ) != l2.phi( i, d ) ) phi_differences++;
        if( phi_differences != 0 )
            problems.push_back( prefix + std::to_string( phi_differences ) + " phi images differ" );

        const auto h1_infos = leafElementInfos( h1 );
        const auto l2_infos = leafElementInfos( l2 );
        bool same = h1_infos.size() == l2_infos.size();
        for( size_t n = 0; same and n < h1_infos.size(); n++ )
            same = h1_infos.at( n ).leaf == l2_infos.at( n ).leaf and keyOf( h1_infos.at( n ) ) == keyOf( l2_infos.at( n ) );
        if( not same ) problems.push_back( prefix + "leaf element infos differ" );
        return problems;
    } );
}

TEST_CASE( "Per patch tensor cell supports of the H1 and L2 level spaces, against the level knots",
           "[amr_queries][multipatch]" )
{
    checkEachFixture( []( const std::string& prefix, const Layout& layout, const size_t degree ) {
        const auto nsd = build( layout, degree );
        std::vector<std::string> problems;
        for( size_t level = 0; level <= layout.marks.size(); level++ )
            for( size_t p = 0; p < layout.patches.size(); p++ )
            {
                const std::string where = prefix + "patch " + std::to_string( p ) + " level " + std::to_string( level );
                const auto h1 = supportProblems( where + " H1, ",
                                                 patchLevelTensorCellSupports( nsd->H1_ss, p, level ),
                                                 levelSupports( layout.patches.at( p ), level, degree ) );
                const auto l2 = supportProblems( where + " L2, ",
                                                 patchLevelTensorCellSupports( nsd->L2_ss, p, level ),
                                                 levelSupports( layout.patches.at( p ), level, degree - 1 ) );
                problems.insert( problems.end(), h1.begin(), h1.end() );
                problems.insert( problems.end(), l2.begin(), l2.end() );
            }
        return problems;
    } );
}

TEST_CASE( "Per patch active functions follow multipatch support containment", "[amr_queries][multipatch]" )
{
    checkEachFixture( []( const std::string& prefix, const Layout& layout, const size_t degree ) {
        const auto nsd = build( layout, degree );
        std::vector<std::string> problems = activeProblems( prefix + "H1, ", nsd->H1_ss, layout, degree, true );
        const auto l2 = activeProblems( prefix + "L2, ", nsd->L2_ss, layout, degree - 1, false );
        problems.insert( problems.end(), l2.begin(), l2.end() );
        return problems;
    } );
}

TEST_CASE( "Multipatch geometry reproduces each patch map on every leaf", "[amr_queries][multipatch]" )
{
    checkEachFixture( []( const std::string& prefix, const Layout& layout, const size_t degree ) {
        return geometryProblems( prefix, *build( layout, degree ), layout );
    } );
}

TEST_CASE( "Multipatch construction refuses malformed input", "[amr_queries][multipatch]" )
{
    const Layout& base = side_by_side;
    const size_t degree = 2;
    const auto attempt = [&]( std::vector<api::PatchGeometry> patches,
                              std::vector<std::pair<api::PatchSideRef, api::PatchSideRef>> gluing,
                              std::vector<std::vector<api::PatchElement>> marks ) {
        return [=]() { api::NavierStokesHierarchicalMultiPatchDiscretization( patches, gluing, marks ); };
    };
    const std::vector<api::PatchGeometry> geometries = patchGeometries( base, degree );
    CHECK_NOTHROW( attempt( geometries, base.gluing, base.marks )() );

    SECTION( "patch geometries that disagree on the interface" )
    {
        auto patches = geometries;
        patches.at( 1 ).control_points.row( 0 ).array() += 1e-6;
        CHECK( contains( refusal<std::invalid_argument>( attempt( patches, base.gluing, base.marks ) ),
                         "disagree on a shared H1 coefficient" ) );
    }
    SECTION( "a scaled interface" )
    {
        Layout scaled = base;
        scaled.patches.at( 1 ).breaks_t = { 0, 2, 6 };
        CHECK( contains( refusal<std::invalid_argument>(
                             attempt( patchGeometries( scaled, degree ), scaled.gluing, scaled.marks ) ),
                         "Strong multipatch coupling" ) );
    }
    SECTION( "a side named twice" )
    {
        auto gluing = base.gluing;
        gluing.push_back( { { 0, PatchSide::S1 }, { 1, PatchSide::S1 } } );
        CHECK( contains( refusal<std::invalid_argument>( attempt( geometries, gluing, base.marks ) ), "more than once" ) );
        CHECK( contains( refusal<std::invalid_argument>(
                             attempt( geometries, { { { 0, PatchSide::S1 }, { 0, PatchSide::S1 } } }, base.marks ) ),
                         "more than once" ) );
    }
    SECTION( "a gluing patch outside the discretization" )
    {
        CHECK( contains( refusal<std::out_of_range>(
                             attempt( geometries, { { { 0, PatchSide::S1 }, { 2, PatchSide::S0 } } }, base.marks ) ),
                         "Gluing names a patch" ) );
    }
    SECTION( "a mark outside the discretization" )
    {
        CHECK( contains( refusal<std::out_of_range>( attempt( geometries, base.gluing, { { { 2, 0, 0 } } } ) ),
                         "names a patch outside" ) );
        CHECK( contains( refusal<std::out_of_range>( attempt( geometries, base.gluing, { { { 0, 2, 0 } } } ) ),
                         "outside its patch level" ) );
        CHECK( contains( refusal<std::out_of_range>( attempt( geometries, base.gluing, { { { 0, 0, 2 } } } ) ),
                         "outside its patch level" ) );
        CHECK( contains( refusal<std::out_of_range>(
                             attempt( geometries, base.gluing, { { { 0, 1, 0 } }, { { 0, 4, 0 } } } ) ),
                         "outside its patch level" ) );
    }
    SECTION( "a mark that is not a leaf" )
    {
        CHECK( contains( refusal<std::runtime_error>(
                             attempt( geometries, base.gluing, { { { 0, 1, 0 } }, { { 0, 0, 0 } } } ) ),
                         "isn't active" ) );
        CHECK( contains( refusal<std::runtime_error>( attempt( geometries, base.gluing, { { { 0, 1, 0 }, { 0, 1, 0 } } } ) ),
                         "isn't active" ) );
    }
    SECTION( "non-finite control points" )
    {
        for( const double bad : { std::nan( "" ), std::numeric_limits<double>::infinity() } )
        {
            CAPTURE( bad );
            auto patches = geometries;
            patches.at( 0 ).control_points( 1, 0 ) = bad;
            CHECK( contains( refusal<std::invalid_argument>( attempt( patches, base.gluing, base.marks ) ), "finite" ) );
        }
    }
    SECTION( "control point shapes" )
    {
        auto columns = geometries;
        columns.at( 0 ).control_points.conservativeResize( Eigen::NoChange, columns.at( 0 ).control_points.cols() - 1 );
        CHECK( contains( refusal<std::invalid_argument>( attempt( columns, base.gluing, base.marks ) ),
                         "control point count" ) );

        auto rows = geometries;
        rows.at( 1 ).control_points.conservativeResize( 3, Eigen::NoChange );
        rows.at( 1 ).control_points.row( 2 ).setZero();
        CHECK( contains( refusal<std::invalid_argument>( attempt( rows, base.gluing, base.marks ) ),
                         "same spatial dimension" ) );

        auto empty = geometries;
        for( api::PatchGeometry& g : empty ) g.control_points.resize( 0, g.control_points.cols() );
        CHECK( contains( refusal<std::invalid_argument>( attempt( empty, base.gluing, base.marks ) ),
                         "at least one spatial coordinate" ) );

        CHECK( contains( refusal<std::invalid_argument>( attempt( {}, {}, {} ) ), "at least one patch" ) );
    }
}

TEST_CASE( "Shared H1 coefficients agree to 1e-10 max( 1, |a|, |b| ), absolute and relative",
           "[amr_queries][multipatch]" )
{
    // The interface at x = x_i: patch 0's shared x coefficients are x_i exactly, so shifting patch 1 by delta in x
    // makes every shared x coefficient differ by delta.
    for( const auto& [x_i, threshold] : { std::pair{ 0.0, 1e-10 }, std::pair{ 1000.0, 1e-7 } } )
        for( const size_t degree : fixture_degrees )
            for( const double factor : { 0.9, 1.1 } )
            {
                CAPTURE( x_i, degree, factor );
                Layout layout = side_by_side;
                layout.patches.at( 0 ).phi = { x_i - 2, 1, 0, 0, 0, 1 };
                layout.patches.at( 1 ).phi = { x_i, 1, 0, 0, 0, 1 };
                std::vector<api::PatchGeometry> patches;
                for( const PatchLayout& patch : layout.patches ) patches.push_back( affineGeometry( patch, degree ) );
                patches.at( 1 ).control_points.row( 0 ).array() += factor * threshold;

                const std::string outcome = refusal<std::invalid_argument>( [&]() {
                    api::NavierStokesHierarchicalMultiPatchDiscretization( patches, layout.gluing, layout.marks );
                } );
                if( factor < 1 )
                    CHECK( outcome == "no exception" );
                else
                    CHECK( contains( outcome, "disagree on a shared H1 coefficient" ) );
            }
}

TEST_CASE( "Per patch basis queries refuse a patch or level outside the space", "[amr_queries][multipatch]" )
{
    const auto nsd = build( side_by_side, 2 );
    const size_t n_levels = side_by_side.marks.size() + 1;
    for( const HierarchicalMultiPatchSplineSpace* ss : { &nsd->H1_ss, &nsd->L2_ss } )
    {
        CHECK_NOTHROW( patchLevelActiveFunctions( *ss, 1, n_levels - 1 ) );
        CHECK_NOTHROW( patchLevelTensorCellSupports( *ss, 1, n_levels - 1 ) );
        CHECK_THROWS_AS( patchLevelActiveFunctions( *ss, 2, 0 ), std::out_of_range );
        CHECK_THROWS_AS( patchLevelActiveFunctions( *ss, 0, n_levels ), std::out_of_range );
        CHECK_THROWS_AS( patchLevelTensorCellSupports( *ss, 2, 0 ), std::out_of_range );
        CHECK_THROWS_AS( patchLevelTensorCellSupports( *ss, 0, n_levels ), std::out_of_range );
    }
}

TEST_CASE( "On one patch the multipatch queries agree with the single patch ones", "[amr_queries][multipatch]" )
{
    // A cross-check only: the oracles above, not this agreement, are the evidence.
    const Layout one{ "one patch", { { { 0, 1, 2 }, { 0, 1 }, identity } }, {}, { { { 0, 1, 0 } }, { { 0, 2, 0 }, { 0, 2, 1 } } } };
    for( const size_t degree : fixture_degrees )
    {
        CAPTURE( degree );
        const auto multi = build( one, degree );
        const api::PatchGeometry g = patchGeometry( one.patches.front(), degree );
        std::vector<std::vector<std::pair<size_t, size_t>>> marks;
        for( const auto& level : one.marks )
        {
            marks.emplace_back();
            for( const auto& [p, i, j] : level ) marks.back().emplace_back( i, j );
        }
        const api::NavierStokesHierarchicalDiscretization single( g.kv_s, g.kv_t, degree, degree, g.control_points, marks );

        std::multiset<Index> multi_leaves;
        for( const auto& info : leafElementInfos( multi->H1_ss.basisComplex().parametricAtlas().cmap() ) )
        {
            CHECK( info.patch == 0 );
            multi_leaves.insert( { info.level, info.indices.at( 0 ), info.indices.at( 1 ) } );
        }
        std::multiset<Index> single_leaves;
        for( const auto& info : leafElementInfos( single.H1_ss.basisComplex().parametricAtlas().cmap() ) )
            single_leaves.insert( { info.level, info.indices.at( 0 ), info.indices.at( 1 ) } );
        CHECK( multi_leaves == single_leaves );

        for( size_t level = 0; level <= one.marks.size(); level++ )
        {
            CAPTURE( level );
            CHECK( patchLevelActiveFunctions( multi->H1_ss, 0, level ) == single.H1_ss.activeFunctions().at( level ) );
            CHECK( patchLevelActiveFunctions( multi->L2_ss, 0, level ) == single.L2_ss.activeFunctions().at( level ) );
            CHECK( patchLevelTensorCellSupports( multi->L2_ss, 0, level ) ==
                   tensorCellSupports( *single.L2_ss.refinementLevels().at( level ) ) );
        }
    }
}
