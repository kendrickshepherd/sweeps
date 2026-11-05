// Item 1, steps 4 to 6 (outstanding_issues.txt): algebraic correctness of
// asymmetric THB leaf selection across a same-mesh multipatch interface.
#include <catch2/catch_test_macros.hpp>
#include <HierarchicalMultiPatchSplineSpace.hpp>
#include <HierarchicalMultiPatchCombinatorialMap.hpp>
#include <KnotVector.hpp>
#include <MultiPatchSplineFactory.hpp>
#include <TraceExtraction.hpp>
#include <TraceMesh.hpp>
#include <VectorConformingHierarchicalMultiPatchSplineSpace.hpp>
#include <MultiPatchSplineSpace.hpp>
#include <ParentPoint.hpp>
#include <SideCoordinateTransform.hpp>
#include <SplineSpaceEvaluator.hpp>
#include <TPSplineSpace.hpp>
#include <CombinatorialMapMethods.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace basis;
using namespace topology;

namespace
{
    using InternalConnectionsMap = MultiPatchCombinatorialMap::InternalConnectionsMap;
    using ConstituentSide = MultiPatchCombinatorialMap::ConstituentSide;
    using TPPermutation = MultiPatchCombinatorialMap::TPPermutation;
    using LevelFunction = std::pair<size_t, size_t>;
    using TPIndex = std::array<size_t, 2>;
    using Box = std::array<std::array<double, 2>, 2>;

    constexpr double geom_tol = 1e-12;
    constexpr double coeff_tol = 1e-10;

    std::shared_ptr<const TPSplineSpace> makePatch( const SmallVector<KnotVector, 3>& kvs,
                                                    const SmallVector<size_t, 3>& degrees )
    {
        return std::make_shared<const TPSplineSpace>( buildBSpline( kvs, degrees ) );
    }

    Cell patchTopCellAt( const MultiPatchCombinatorialMap& cmap, const size_t patch_id, const TPIndex& element )
    {
        const FullyUnflattenedDart unflat( { Dart( element[0] ), Dart( element[1] ) } );
        return Cell( cmap.toGlobalDart( patch_id, flattenFull( *cmap.constituents().at( patch_id ), unflat ) ),
                     cmap.dim() );
    }

    std::vector<Cell> topCells( const CombinatorialMap& cmap )
    {
        std::vector<Cell> out;
        iterateCellsWhile( cmap, cmap.dim(), [&]( const Cell& c ) {
            out.push_back( c );
            return true;
        } );
        return out;
    }

    std::vector<Cell> topCellsForPatch( const MultiPatchCombinatorialMap& cmap, const size_t patch_id )
    {
        std::vector<Cell> out;
        for( const Cell& c : topCells( cmap ) )
            if( cmap.toLocalDart( c.dart() ).first == patch_id ) out.push_back( c );
        return out;
    }

    // Patch 0's s-max side glued to patch 1's s-min side.
    InternalConnectionsMap sMaxToSMin()
    {
        const ConstituentSide first{ 0, ElementSide( 0, false ).sideId() };
        const ConstituentSide second{ 1, ElementSide( 0, true ).sideId() };
        return { { first, { TPPermutation::Flip1d, second } }, { second, { TPPermutation::Flip1d, first } } };
    }

    const MultiPatchCombinatorialMap& levelCMap( const MultiPatchSplineSpace& ss )
    {
        return dynamic_cast<const MultiPatchCombinatorialMap&>( ss.basisComplex().parametricAtlas().cmap() );
    }

    // Requested leaves, their breakpoints per level and direction (shared by
    // both patches), and the hand-derived patch-local coefficient of a coarse
    // shared function in a fine function of the same component.
    struct Fixture
    {
        std::vector<std::vector<Cell>> leaves;
        std::vector<std::array<std::vector<double>, 2>> breakpoints;
        std::function<double( const TPIndex& coarse,
                              const TPIndex& fine,
                              const TPIndex& coarse_lengths,
                              const TPIndex& fine_lengths )>
            shared_truncation;
    };

    struct PatchFunction
    {
        size_t local;
        double sign;
    };

    // Everything is normalized to patch-local tensor-product functions: a row
    // times its sign is the unsigned patch-local row.
    struct HierarchyView
    {
        std::string name;
        const SplineSpace* hier = nullptr;
        const HierarchicalMultiPatchCombinatorialMap* cmap = nullptr;
        std::vector<const SplineSpace*> levels;
        std::vector<std::pair<size_t, FunctionId>> level_of;
        std::map<std::pair<size_t, size_t>, double> hier_sign;                  // (patch, hierarchical fid)
        std::vector<std::vector<std::map<size_t, PatchFunction>>> level_local;  // [level][patch][level fid]
        std::vector<std::vector<std::vector<size_t>>> offsets;                  // [level][patch], ncomp + 1
        std::vector<std::vector<std::vector<TPIndex>>> lengths;                 // [level][patch][component]
    };

    size_t globalOf( const FunctionId& f ) { return f.id(); }
    size_t globalOf( const std::pair<FunctionId, bool>& f ) { return f.first.id(); }
    double signOf( const FunctionId& ) { return 1.0; }
    double signOf( const std::pair<FunctionId, bool>& f ) { return f.second ? 1.0 : -1.0; }

    template <typename Entry>
    std::map<size_t, PatchFunction> invertPatchMap( const std::vector<Entry>& patch_map )
    {
        std::map<size_t, PatchFunction> out;
        for( size_t local = 0; local < patch_map.size(); local++ )
        {
            const bool inserted =
                out.emplace( globalOf( patch_map.at( local ) ), PatchFunction{ local, signOf( patch_map.at( local ) ) } )
                    .second;
            REQUIRE( inserted ); // no function appears twice on one patch in these fixtures
        }
        return out;
    }

    TPIndex lengthsOf( const TPSplineSpace& tp ) { return { tp.source().numFunctions(), tp.line().numFunctions() }; }

    template <typename HierarchicalSpace>
    HierarchyView commonView( const std::string& name, const HierarchicalSpace& ss )
    {
        HierarchyView v;
        v.name = name;
        v.hier = &ss;
        v.cmap = &dynamic_cast<const HierarchicalMultiPatchCombinatorialMap&>( ss.basisComplex().parametricAtlas().cmap() );
        for( size_t patch = 0; patch < ss.functionIdMap().size(); patch++ )
            for( const auto& [global, entry] : invertPatchMap( ss.functionIdMap().at( patch ) ) )
                v.hier_sign[{ patch, global }] = entry.sign;
        for( const auto& level : ss.refinementLevels() )
        {
            v.levels.push_back( level.get() );
            v.level_local.emplace_back();
            for( const auto& patch_map : level->functionIdMap() ) v.level_local.back().push_back( invertPatchMap( patch_map ) );
        }
        return v;
    }

    HierarchyView viewOf( const std::string& name, const HierarchicalMultiPatchSplineSpace& ss )
    {
        HierarchyView v = commonView( name, ss );
        v.level_of = fidsToRefinementLevelFids( ss );
        for( const auto& level : ss.refinementLevels() )
        {
            v.offsets.emplace_back();
            v.lengths.emplace_back();
            for( const auto& tp : level->subSpaces() )
            {
                v.offsets.back().push_back( { 0, tp->numFunctions() } );
                v.lengths.back().push_back( { lengthsOf( *tp ) } );
            }
        }
        return v;
    }

    HierarchyView viewOf( const std::string& name, const VectorConformingHierarchicalMultiPatchSplineSpace& ss )
    {
        HierarchyView v = commonView( name, ss );
        v.level_of = vectorFidsToRefinementLevelFids( ss );
        for( const auto& level : ss.refinementLevels() )
        {
            v.offsets.emplace_back();
            v.lengths.emplace_back();
            for( const auto& vector_tp : level->subSpaces() )
            {
                std::vector<size_t> offsets{ 0 };
                std::vector<TPIndex> lengths;
                for( const auto& component : vector_tp->scalarTPBases() )
                {
                    offsets.push_back( offsets.back() + component->numFunctions() );
                    lengths.push_back( lengthsOf( *component ) );
                }
                v.offsets.back().push_back( offsets );
                v.lengths.back().push_back( lengths );
            }
        }
        return v;
    }

    const PatchFunction& patchFunction( const HierarchyView& v, const size_t level, const size_t patch, const size_t fid )
    {
        const auto& patch_map = v.level_local.at( level ).at( patch );
        const auto it = patch_map.find( fid );
        REQUIRE( it != patch_map.end() );
        return it->second;
    }

    size_t componentOf( const HierarchyView& v, const size_t level, const size_t patch, const size_t fid )
    {
        const auto& offsets = v.offsets.at( level ).at( patch );
        const size_t local = patchFunction( v, level, patch, fid ).local;
        return static_cast<size_t>( std::upper_bound( offsets.begin(), offsets.end(), local ) - offsets.begin() ) - 1;
    }

    // (s, t) index within the component, s fastest.
    TPIndex tpIndexOf( const HierarchyView& v, const size_t level, const size_t patch, const size_t fid )
    {
        const size_t component = componentOf( v, level, patch, fid );
        const size_t local = patchFunction( v, level, patch, fid ).local - v.offsets.at( level ).at( patch ).at( component );
        const TPIndex& lengths = v.lengths.at( level ).at( patch ).at( component );
        return { local % lengths[0], local / lengths[0] };
    }

    struct LevelCell
    {
        size_t level;
        Cell cell;
        size_t patch;
        Box box;
    };

    LevelCell levelCellInfo( const HierarchyView& v, const Fixture& fx, const size_t level, const Cell& level_cell )
    {
        const MultiPatchCombinatorialMap& level_cmap = *v.cmap->refinementLevels().at( level );
        const auto [patch, local_d] = level_cmap.toLocalDart( level_cell.dart() );
        const FullyUnflattenedDart unflat = unflattenFull( *level_cmap.constituents().at( patch ), local_d );
        Box box;
        for( size_t dir = 0; dir < 2; dir++ )
        {
            const size_t e = unflat.unflat_darts.at( dir ).id();
            box[dir] = { fx.breakpoints.at( level )[dir].at( e ), fx.breakpoints.at( level )[dir].at( e + 1 ) };
        }
        return { level, level_cell, patch, box };
    }

    double measure( const Box& b ) { return ( b[0][1] - b[0][0] ) * ( b[1][1] - b[1][0] ); }

    double overlap( const Box& a, const Box& b )
    {
        double out = 1.0;
        for( size_t dir = 0; dir < 2; dir++ )
            out *= std::max( 0.0, std::min( a[dir][1], b[dir][1] ) - std::max( a[dir][0], b[dir][0] ) );
        return out;
    }

    // Omega^m on a patch is the union of requested leaf boxes of level >= m.
    // A cell box must lie entirely inside or entirely outside it.
    bool insideOmega( const std::vector<LevelCell>& leaves, const LevelCell& c, const size_t m )
    {
        double covered = 0.0;
        for( const LevelCell& leaf : leaves )
            if( leaf.patch == c.patch and leaf.level >= m ) covered += overlap( leaf.box, c.box );
        const bool inside = std::abs( covered - measure( c.box ) ) < geom_tol;
        const bool outside = std::abs( covered ) < geom_tol;
        INFO( "level " << c.level << " cell on patch " << c.patch << " against Omega^" << m );
        REQUIRE( ( inside or outside ) );
        return inside;
    }

    // Full THB support containment over all level cells: active iff
    // supp b in Omega^l and supp b not in Omega^(l+1).
    std::set<LevelFunction> expectedActive( const HierarchyView& v, const Fixture& fx, const std::vector<LevelCell>& leaves )
    {
        std::set<LevelFunction> out;
        for( size_t level = 0; level < v.levels.size(); level++ )
        {
            std::map<size_t, std::vector<std::pair<bool, bool>>> support;
            for( const Cell& c : topCells( *v.cmap->refinementLevels().at( level ) ) )
            {
                const LevelCell info = levelCellInfo( v, fx, level, c );
                const std::pair<bool, bool> inside{ insideOmega( leaves, info, level ),
                                                    insideOmega( leaves, info, level + 1 ) };
                for( const FunctionId& f : v.levels.at( level )->connectivity( c ) ) support[f.id()].push_back( inside );
            }
            REQUIRE( support.size() == v.levels.at( level )->numFunctions() );
            for( const auto& [f, cells] : support )
            {
                const bool in_level = std::all_of( cells.begin(), cells.end(), []( const auto& pr ) { return pr.first; } );
                const bool in_finer = std::all_of( cells.begin(), cells.end(), []( const auto& pr ) { return pr.second; } );
                if( in_level and not in_finer ) out.insert( { level, f } );
            }
        }
        return out;
    }

    // Level functions supported on both patches, from level connectivity alone.
    std::vector<std::set<size_t>> sharedFunctions( const HierarchyView& v )
    {
        std::vector<std::set<size_t>> out;
        for( size_t level = 0; level < v.levels.size(); level++ )
        {
            const MultiPatchCombinatorialMap& level_cmap = *v.cmap->refinementLevels().at( level );
            std::map<size_t, std::set<size_t>> patches;
            for( const Cell& c : topCells( level_cmap ) )
                for( const FunctionId& f : v.levels.at( level )->connectivity( c ) )
                    patches[f.id()].insert( level_cmap.toLocalDart( c.dart() ).first );
            out.emplace_back();
            for( const auto& [f, ps] : patches )
                if( ps.size() > 1 ) out.back().insert( f );
        }
        return out;
    }

    Eigen::MatrixXd signedRows( const Eigen::MatrixXd& rows, const std::vector<double>& signs )
    {
        Eigen::VectorXd s( static_cast<Eigen::Index>( signs.size() ) );
        for( size_t i = 0; i < signs.size(); i++ ) s( static_cast<Eigen::Index>( i ) ) = signs.at( i );
        return s.asDiagonal() * rows;
    }

    size_t indexIn( const std::vector<FunctionId>& conn, const size_t fid )
    {
        for( size_t i = 0; i < conn.size(); i++ )
            if( static_cast<size_t>( conn.at( i ).id() ) == fid ) return i;
        return conn.size();
    }

    bool hasDuplicates( const std::vector<FunctionId>& conn )
    {
        std::set<size_t> ids;
        for( const FunctionId& f : conn ) ids.insert( f.id() );
        return ids.size() != conn.size();
    }

    // Requested leaves; they must be the effective leaves and tile both patches.
    std::vector<LevelCell> requestedLeaves( const HierarchyView& v, const Fixture& fx )
    {
        for( size_t level = 0; level < v.levels.size(); level++ )
            REQUIRE( &v.levels.at( level )->basisComplex().parametricAtlas().cmap() ==
                     v.cmap->refinementLevels().at( level ).get() );

        std::vector<LevelCell> leaves;
        const std::vector<std::vector<Cell>> effective = leafElements( *v.cmap );
        REQUIRE( effective.size() == fx.leaves.size() );
        for( size_t level = 0; level < fx.leaves.size(); level++ )
        {
            std::set<size_t> requested_darts, effective_darts;
            for( const Cell& c : fx.leaves.at( level ) )
            {
                requested_darts.insert( c.dart().id() );
                leaves.push_back( levelCellInfo( v, fx, level, c ) );
            }
            for( const Cell& c : effective.at( level ) ) effective_darts.insert( c.dart().id() );
            REQUIRE( requested_darts == effective_darts );
        }

        const auto& bp = fx.breakpoints.front();
        const double patch_measure = ( bp[0].back() - bp[0].front() ) * ( bp[1].back() - bp[1].front() );
        for( size_t patch = 0; patch < 2; patch++ )
        {
            double covered = 0.0;
            for( size_t a = 0; a < leaves.size(); a++ )
            {
                if( leaves.at( a ).patch != patch ) continue;
                covered += measure( leaves.at( a ).box );
                for( size_t b = a + 1; b < leaves.size(); b++ )
                    if( leaves.at( b ).patch == patch )
                        REQUIRE( overlap( leaves.at( a ).box, leaves.at( b ).box ) < geom_tol );
            }
            REQUIRE( std::abs( covered - patch_measure ) < geom_tol );
        }
        return leaves;
    }

    // Two levels, patch 1 entirely coarse.  Checks: (1) coarse shared functions
    // active, (2) fine shared functions inactive, (3) truncation on fine
    // leaves, (4) rows on coarse leaves, (5) per-component partition of unity.
    void checkSharedInterfaceRefinement( const HierarchyView& v, const Fixture& fx )
    {
        INFO( v.name );
        REQUIRE( v.levels.size() == 2 );
        const std::vector<LevelCell> leaves = requestedLeaves( v, fx );

        // Oracle against the space's (level, function) pairs, as multisets.
        const std::set<LevelFunction> expected = expectedActive( v, fx, leaves );
        std::multiset<LevelFunction> actual;
        for( const auto& [level, fid] : v.level_of ) actual.insert( { level, fid.id() } );
        CHECK( v.level_of.size() == v.hier->numFunctions() );
        CHECK( std::set<LevelFunction>( actual.begin(), actual.end() ).size() == actual.size() );
        CHECK( actual == std::multiset<LevelFunction>( expected.begin(), expected.end() ) );

        // (1), (2).  Fixture-specific: patch 1 is coarse throughout.
        const std::vector<std::set<size_t>> shared = sharedFunctions( v );
        REQUIRE_FALSE( shared.at( 0 ).empty() );
        REQUIRE_FALSE( shared.at( 1 ).empty() );
        for( const size_t f : shared.at( 0 ) ) CHECK( expected.contains( { 0, f } ) );
        for( const size_t f : shared.at( 1 ) ) CHECK_FALSE( expected.contains( { 1, f } ) );

        size_t coarse_leaves = 0, fine_leaves = 0, truncated_zeros = 0, coarse_shared_rows = 0, shared_sums = 0;
        std::map<std::array<size_t, 3>, double> coefficients;
        for( const Cell& c : topCells( *v.cmap ) )
        {
            const auto [level, level_d] = v.cmap->unrefinedAncestorDartOfCell( c );
            const LevelCell info = levelCellInfo( v, fx, level, Cell( level_d, c.dim() ) );
            const size_t patch = info.patch;
            INFO( "leaf on patch " << patch << ", level " << level << ", box [" << info.box[0][0] << ", "
                                   << info.box[0][1] << "] x [" << info.box[1][0] << ", " << info.box[1][1] << "]" );

            const std::vector<FunctionId> hconn = v.hier->connectivity( c );
            const std::vector<FunctionId> lconn = v.levels.at( level )->connectivity( info.cell );
            REQUIRE_FALSE( hasDuplicates( hconn ) );
            REQUIRE_FALSE( hasDuplicates( lconn ) );

            std::vector<double> hsigns, lsigns;
            std::vector<size_t> hlevel, hfid, hcomp, lcomp;
            for( const FunctionId& h : hconn )
            {
                const auto [h_level, h_fid] = v.level_of.at( h.id() );
                const auto it = v.hier_sign.find( { patch, h.id() } );
                REQUIRE( it != v.hier_sign.end() );
                hsigns.push_back( it->second );
                hlevel.push_back( h_level );
                hfid.push_back( h_fid.id() );
                hcomp.push_back( componentOf( v, h_level, patch, h_fid.id() ) );
                // The hierarchical sign is the level function's sign on this patch.
                CHECK( it->second == patchFunction( v, h_level, patch, h_fid.id() ).sign );
            }
            for( const FunctionId& f : lconn )
            {
                lsigns.push_back( patchFunction( v, level, patch, f.id() ).sign );
                lcomp.push_back( componentOf( v, level, patch, f.id() ) );
            }

            const Eigen::MatrixXd H = signedRows( v.hier->extractionOperator( c ), hsigns );
            const Eigen::MatrixXd L = signedRows( v.levels.at( level )->extractionOperator( info.cell ), lsigns );
            REQUIRE( H.rows() == static_cast<Eigen::Index>( hconn.size() ) );
            REQUIRE( L.rows() == static_cast<Eigen::Index>( lconn.size() ) );
            REQUIRE( H.cols() == L.cols() );

            // Bernstein column block of each component, read from the level rows.
            std::vector<size_t> column_component( static_cast<size_t>( L.cols() ) );
            for( Eigen::Index j = 0; j < L.cols(); j++ )
            {
                std::set<size_t> comps;
                for( Eigen::Index i = 0; i < L.rows(); i++ )
                    if( std::abs( L( i, j ) ) > geom_tol ) comps.insert( lcomp.at( static_cast<size_t>( i ) ) );
                REQUIRE( comps.size() == 1 );
                column_component.at( static_cast<size_t>( j ) ) = *comps.begin();
            }

            // (5) Per component, the rows sum to the all-ones Bernstein vector of
            // that component's block: every column sum is one.
            const size_t num_components = v.offsets.at( level ).at( patch ).size() - 1;
            for( size_t comp = 0; comp < num_components; comp++ )
            {
                Eigen::RowVectorXd sum = Eigen::RowVectorXd::Zero( H.cols() );
                for( size_t i = 0; i < hconn.size(); i++ )
                    if( hcomp.at( i ) == comp ) sum += H.row( static_cast<Eigen::Index>( i ) );
                for( Eigen::Index j = 0; j < H.cols(); j++ )
                    CHECK( std::abs( sum( j ) - ( column_component.at( static_cast<size_t>( j ) ) == comp ? 1.0 : 0.0 ) ) <
                           geom_tol );
            }

            if( level == 0 )
            {
                // (4) Coarse leaf: the rows are the level-0 rows, aligned by level
                // function and normalized by sign.
                coarse_leaves++;
                for( size_t i = 0; i < hconn.size(); i++ )
                {
                    REQUIRE( hlevel.at( i ) == 0 );
                    const size_t j = indexIn( lconn, hfid.at( i ) );
                    REQUIRE( j < lconn.size() );
                    CHECK( ( H.row( static_cast<Eigen::Index>( i ) ) - L.row( static_cast<Eigen::Index>( j ) ) )
                               .cwiseAbs()
                               .maxCoeff() < geom_tol );
                }
                CHECK( hconn.size() == lconn.size() );
                continue;
            }

            // (3) Fine leaf.  Solve H = A L; L must be square and of full rank.
            fine_leaves++;
            REQUIRE( L.rows() == L.cols() );
            const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr( L.transpose() );
            REQUIRE( qr.rank() == L.rows() );
            const Eigen::MatrixXd A = qr.solve( H.transpose() ).transpose();
            REQUIRE( ( A * L - H ).cwiseAbs().maxCoeff() < coeff_tol );

            const auto coarse_parent = [&]() {
                for( const Cell& e : topCells( *v.cmap->refinementLevels().at( 0 ) ) )
                {
                    const LevelCell parent = levelCellInfo( v, fx, 0, e );
                    if( parent.patch == patch and std::abs( overlap( parent.box, info.box ) - measure( info.box ) ) < geom_tol )
                        return parent;
                }
                FAIL( "No coarse parent element." );
                return info;
            }();

            bool has_fine_shared = false;
            for( const FunctionId& f : lconn ) has_fine_shared = has_fine_shared or shared.at( 1 ).contains( f.id() );

            for( size_t i = 0; i < hconn.size(); i++ )
            {
                const Eigen::Index ii = static_cast<Eigen::Index>( i );
                if( hlevel.at( i ) == level )
                {
                    const size_t k = indexIn( lconn, hfid.at( i ) );
                    REQUIRE( k < lconn.size() );
                    for( size_t j = 0; j < lconn.size(); j++ )
                        CHECK( std::abs( A( ii, static_cast<Eigen::Index>( j ) ) - ( j == k ? 1.0 : 0.0 ) ) < coeff_tol );
                    continue;
                }

                const bool coarse_shared = shared.at( 0 ).contains( hfid.at( i ) );
                if( coarse_shared ) coarse_shared_rows++;
                if( coarse_shared and not has_fine_shared ) CHECK( H.row( ii ).cwiseAbs().maxCoeff() < geom_tol );
                for( size_t j = 0; j < lconn.size(); j++ )
                {
                    const double a = A( ii, static_cast<Eigen::Index>( j ) );
                    INFO( "coarse level function " << hfid.at( i ) << ", fine level function " << lconn.at( j ).id() );
                    if( expected.contains( { level, lconn.at( j ).id() } ) )
                    {
                        CHECK( std::abs( a ) < coeff_tol );
                        truncated_zeros++;
                    }
                    const auto [it, inserted] = coefficients.emplace(
                        std::array<size_t, 3>{ patch, static_cast<size_t>( hconn.at( i ).id() ), static_cast<size_t>( lconn.at( j ).id() ) },
                        a );
                    if( not inserted ) CHECK( std::abs( it->second - a ) < coeff_tol );

                    if( not coarse_shared ) continue;
                    // Fixture-specific expected values.
                    if( not shared.at( 1 ).contains( lconn.at( j ).id() ) ) CHECK( std::abs( a ) < coeff_tol );
                    const size_t comp = hcomp.at( i );
                    const double exact =
                        lcomp.at( j ) != comp
                            ? 0.0
                            : fx.shared_truncation( tpIndexOf( v, 0, patch, hfid.at( i ) ),
                                                    tpIndexOf( v, level, patch, lconn.at( j ).id() ),
                                                    v.lengths.at( 0 ).at( patch ).at( comp ),
                                                    v.lengths.at( level ).at( patch ).at( comp ) );
                    CHECK( std::abs( a - exact ) < coeff_tol );
                }
            }

            // Secondary: per component, over the complete set of coarse shared
            // functions on the coarse parent, each fine shared column sums to one.
            // Without a fine shared column the coarse shared rows are zero (above)
            // and may be omitted.
            if( not has_fine_shared ) continue;
            const std::vector<FunctionId> parent_conn = v.levels.at( 0 )->connectivity( coarse_parent.cell );
            for( size_t comp = 0; comp < num_components; comp++ )
            {
                std::set<size_t> on_parent, on_leaf;
                for( const FunctionId& f : parent_conn )
                    if( shared.at( 0 ).contains( f.id() ) and componentOf( v, 0, patch, f.id() ) == comp )
                        on_parent.insert( f.id() );
                for( size_t i = 0; i < hconn.size(); i++ )
                    if( hlevel.at( i ) == 0 and hcomp.at( i ) == comp and shared.at( 0 ).contains( hfid.at( i ) ) )
                        on_leaf.insert( hfid.at( i ) );
                CHECK( on_parent == on_leaf );

                for( size_t j = 0; j < lconn.size(); j++ )
                {
                    if( lcomp.at( j ) != comp or not shared.at( 1 ).contains( lconn.at( j ).id() ) ) continue;
                    double sum = 0.0;
                    for( size_t i = 0; i < hconn.size(); i++ )
                        if( on_leaf.contains( hfid.at( i ) ) and hlevel.at( i ) == 0 )
                            sum += A( static_cast<Eigen::Index>( i ), static_cast<Eigen::Index>( j ) );
                    CHECK( std::abs( sum - 1.0 ) < coeff_tol );
                    shared_sums++;
                }
            }
        }

        CHECK( coarse_leaves > 0 );
        CHECK( fine_leaves > 0 );
        CHECK( truncated_zeros > 0 );
        CHECK( coarse_shared_rows > 0 );
        CHECK( shared_sums > 0 );
    }
}

TEST_CASE( "Experiment A: THB activity, truncation and extraction across an asymmetric interface",
           "[shared_interface_refinement]" )
{
    const double ptol = 1e-10;
    const KnotVector coarse_s( { 0, 0, 0, 1, 2, 2, 2 }, ptol );
    const KnotVector fine_s( { 0, 0, 0, 0.5, 1, 1.5, 2, 2, 2 }, ptol );
    const KnotVector t( { 0, 0, 0, 1, 1, 1 }, ptol );

    const auto coarse_patch = makePatch( { coarse_s, t }, { 2, 2 } );
    const auto fine_patch = makePatch( { fine_s, t }, { 2, 2 } );
    auto coarse_h1 = std::make_shared<const MultiPatchSplineSpace>(
        buildH1MultiPatchSplineSpace( { coarse_patch, coarse_patch }, sMaxToSMin() ) );
    auto fine_h1 = std::make_shared<const MultiPatchSplineSpace>(
        buildH1MultiPatchSplineSpace( { fine_patch, fine_patch }, sMaxToSMin() ) );

    // Patch 0 refined on [1, 2] x [0, 1], at its interface; patch 1 coarse.
    Fixture fx;
    fx.leaves = { { patchTopCellAt( levelCMap( *coarse_h1 ), 0, { 0, 0 } ),
                    patchTopCellAt( levelCMap( *coarse_h1 ), 1, { 0, 0 } ),
                    patchTopCellAt( levelCMap( *coarse_h1 ), 1, { 1, 0 } ) },
                  { patchTopCellAt( levelCMap( *fine_h1 ), 0, { 2, 0 } ),
                    patchTopCellAt( levelCMap( *fine_h1 ), 0, { 3, 0 } ) } };
    fx.breakpoints = { { std::vector<double>{ 0, 1, 2 }, std::vector<double>{ 0, 1 } },
                       { std::vector<double>{ 0, 0.5, 1, 1.5, 2 }, std::vector<double>{ 0, 1 } } };
    // The last coarse s function is half the second-to-last fine one plus the
    // last; the former is active, so truncation leaves exactly the last.  t is
    // not refined.
    fx.shared_truncation = []( const TPIndex& coarse, const TPIndex& fine, const TPIndex& coarse_lengths,
                               const TPIndex& fine_lengths ) {
        CHECK( coarse[0] + 1 == coarse_lengths[0] );
        return ( fine[0] + 1 == fine_lengths[0] and fine[1] == coarse[1] ) ? 1.0 : 0.0;
    };

    const HierarchicalMultiPatchSplineSpace h1 = buildHierarchicalSplineSpace( { coarse_h1, fine_h1 }, fx.leaves );
    const VectorConformingHierarchicalMultiPatchSplineSpace hdiv = buildHDivHierarchicalMultiPatchSplineSpace( h1 );
    const VectorConformingHierarchicalMultiPatchSplineSpace hcurl = buildHCurlHierarchicalMultiPatchSplineSpace( h1 );

    checkSharedInterfaceRefinement( viewOf( "H1", h1 ), fx );
    checkSharedInterfaceRefinement( viewOf( "H(div)", hdiv ), fx );
    checkSharedInterfaceRefinement( viewOf( "H(curl)", hcurl ), fx );
}

TEST_CASE( "Tangentially refined interface: THB activity, truncation and extraction",
           "[shared_interface_refinement]" )
{
    const double ptol = 1e-10;
    const KnotVector coarse_kv( { 0, 0, 0, 1, 1, 1 }, ptol );
    const KnotVector fine_kv( { 0, 0, 0, 0.5, 1, 1, 1 }, ptol );

    const auto coarse_patch = makePatch( { coarse_kv, coarse_kv }, { 2, 2 } );
    const auto fine_patch = makePatch( { fine_kv, fine_kv }, { 2, 2 } );
    auto coarse_h1 = std::make_shared<const MultiPatchSplineSpace>(
        buildH1MultiPatchSplineSpace( { coarse_patch, coarse_patch }, sMaxToSMin() ) );
    auto fine_h1 = std::make_shared<const MultiPatchSplineSpace>(
        buildH1MultiPatchSplineSpace( { fine_patch, fine_patch }, sMaxToSMin() ) );

    // Patch 0 fine in both directions, patch 1 coarse: one coarse interface
    // side faces two fine ones.
    Fixture fx;
    fx.leaves = { topCellsForPatch( levelCMap( *coarse_h1 ), 1 ), topCellsForPatch( levelCMap( *fine_h1 ), 0 ) };
    fx.breakpoints = { { std::vector<double>{ 0, 1 }, std::vector<double>{ 0, 1 } },
                       { std::vector<double>{ 0, 0.5, 1 }, std::vector<double>{ 0, 0.5, 1 } } };
    // In s only the last fine function survives, as in Experiment A.  In t the
    // coarse function keeps its knot-insertion coefficients at t = 1/2.
    fx.shared_truncation = []( const TPIndex& coarse, const TPIndex& fine, const TPIndex& coarse_lengths,
                               const TPIndex& fine_lengths ) {
        static const std::vector<std::vector<double>> quadratic = { { 1, 0.5, 0, 0 },
                                                                    { 0, 0.5, 0.5, 0 },
                                                                    { 0, 0, 0.5, 1 } };
        static const std::vector<std::vector<double>> linear = { { 1, 0.5, 0 }, { 0, 0.5, 1 } };
        CHECK( coarse[0] + 1 == coarse_lengths[0] );
        const auto& refinement = coarse_lengths[1] == 3 ? quadratic : linear;
        REQUIRE( refinement.size() == coarse_lengths[1] );
        REQUIRE( refinement.front().size() == fine_lengths[1] );
        return fine[0] + 1 == fine_lengths[0] ? refinement.at( coarse[1] ).at( fine[1] ) : 0.0;
    };

    const HierarchicalMultiPatchSplineSpace h1 = buildHierarchicalSplineSpace( { coarse_h1, fine_h1 }, fx.leaves );
    const VectorConformingHierarchicalMultiPatchSplineSpace hdiv = buildHDivHierarchicalMultiPatchSplineSpace( h1 );
    const VectorConformingHierarchicalMultiPatchSplineSpace hcurl = buildHCurlHierarchicalMultiPatchSplineSpace( h1 );

    checkSharedInterfaceRefinement( viewOf( "H1", h1 ), fx );
    checkSharedInterfaceRefinement( viewOf( "H(div)", hdiv ), fx );
    checkSharedInterfaceRefinement( viewOf( "H(curl)", hcurl ), fx );
}

// Step 5: three levels, leaf-level jumps of two (E6), against full support
// containment.
namespace
{
    std::string where( const LevelCell& c )
    {
        std::ostringstream out;
        out << "patch " << c.patch << " level " << c.level << " s in [" << c.box[0][0] << ", " << c.box[0][1] << "]";
        return out.str();
    }

    std::string describe( const HierarchyView& v, const LevelFunction& f )
    {
        std::ostringstream out;
        out << "L" << f.first << " f" << f.second;
        for( size_t patch = 0; patch < v.level_local.at( f.first ).size(); patch++ )
        {
            if( not v.level_local.at( f.first ).at( patch ).contains( f.second ) ) continue;
            const TPIndex tp = tpIndexOf( v, f.first, patch, f.second );
            out << " p" << patch << "c" << componentOf( v, f.first, patch, f.second ) << "(" << tp[0] << "," << tp[1]
                << ")";
        }
        return out.str();
    }

    std::string describe( const HierarchyView& v, const std::set<LevelFunction>& fs )
    {
        std::string out;
        for( const LevelFunction& f : fs ) out += ( out.empty() ? "" : "; " ) + describe( v, f );
        return out;
    }

    std::string joined( const std::vector<std::string>& lines )
    {
        std::string out;
        for( const std::string& line : lines ) out += line + "\n";
        return out;
    }

    void checkActivity( const HierarchyView& v,
                        const std::set<LevelFunction>& expected,
                        std::vector<std::string>& problems )
    {
        std::multiset<LevelFunction> actual;
        for( const auto& [level, fid] : v.level_of ) actual.insert( { level, fid.id() } );
        const std::set<LevelFunction> distinct( actual.begin(), actual.end() );
        if( v.level_of.size() != v.hier->numFunctions() )
            problems.push_back( v.name + ": level map size is not the function count" );
        if( distinct.size() != actual.size() ) problems.push_back( v.name + ": two functions share a level function" );
        std::set<LevelFunction> extra, missing;
        std::set_difference(
            distinct.begin(), distinct.end(), expected.begin(), expected.end(), std::inserter( extra, extra.end() ) );
        std::set_difference( expected.begin(),
                             expected.end(),
                             distinct.begin(),
                             distinct.end(),
                             std::inserter( missing, missing.end() ) );
        if( not extra.empty() ) problems.push_back( v.name + ": active, not expected: " + describe( v, extra ) );
        if( not missing.empty() ) problems.push_back( v.name + ": expected, not active: " + describe( v, missing ) );
    }

    struct LeafRows
    {
        LevelCell info;
        std::vector<FunctionId> conn;
        Eigen::MatrixXd rows;
    };

    // H1 with t unrefined, so every leaf spans t.  On each leaf the rows sum to
    // one; across each s-face inside a patch every function has one trace.
    // Bernstein columns run s fastest.  Returns the number of faces.
    size_t checkH1Conformity( const HierarchyView& v,
                              const Fixture& fx,
                              const TPIndex& degrees,
                              std::vector<std::string>& problems )
    {
        const Eigen::Index ns = static_cast<Eigen::Index>( degrees[0] + 1 );
        const Eigen::Index nt = static_cast<Eigen::Index>( degrees[1] + 1 );
        const auto& t_bp = fx.breakpoints.front()[1];
        REQUIRE( t_bp.size() == 2 );
        std::vector<LeafRows> leaves;
        for( const Cell& c : topCells( *v.cmap ) )
        {
            const auto [level, level_d] = v.cmap->unrefinedAncestorDartOfCell( c );
            LeafRows leaf{ levelCellInfo( v, fx, level, Cell( level_d, c.dim() ) ),
                           v.hier->connectivity( c ),
                           v.hier->extractionOperator( c ) };
            REQUIRE_FALSE( hasDuplicates( leaf.conn ) );
            REQUIRE( leaf.rows.rows() == static_cast<Eigen::Index>( leaf.conn.size() ) );
            REQUIRE( leaf.rows.cols() == ns * nt );
            REQUIRE( std::abs( leaf.info.box[1][0] - t_bp.front() ) < geom_tol );
            REQUIRE( std::abs( leaf.info.box[1][1] - t_bp.back() ) < geom_tol );
            const double pou = ( leaf.rows.colwise().sum().array() - 1.0 ).abs().maxCoeff();
            if( pou > coeff_tol )
                problems.push_back( v.name + ": rows on " + where( leaf.info ) + " miss partition of unity by " +
                                    std::to_string( pou ) );
            leaves.push_back( std::move( leaf ) );
        }

        size_t faces = 0;
        for( const LeafRows& left : leaves )
            for( const LeafRows& right : leaves )
            {
                if( left.info.patch != right.info.patch or
                    std::abs( left.info.box[0][1] - right.info.box[0][0] ) > geom_tol )
                    continue;
                faces++;
                std::set<size_t> fids;
                for( const FunctionId& f : left.conn ) fids.insert( f.id() );
                for( const FunctionId& f : right.conn ) fids.insert( f.id() );
                for( const size_t f : fids )
                {
                    const size_t i = indexIn( left.conn, f );
                    const size_t j = indexIn( right.conn, f );
                    double jump = 0.0;
                    for( Eigen::Index tb = 0; tb < nt; tb++ )
                    {
                        const double a =
                            i < left.conn.size() ? left.rows( static_cast<Eigen::Index>( i ), ns - 1 + ns * tb ) : 0.0;
                        const double b =
                            j < right.conn.size() ? right.rows( static_cast<Eigen::Index>( j ), ns * tb ) : 0.0;
                        jump = std::max( jump, std::abs( a - b ) );
                    }
                    if( jump <= coeff_tol ) continue;
                    const auto& [level, fid] = v.level_of.at( f );
                    problems.push_back( v.name + ": trace of " + describe( v, LevelFunction{ level, fid.id() } ) +
                                        " jumps by " + std::to_string( jump ) + " from " + where( left.info ) + " to " +
                                        where( right.info ) );
                }
            }
        return faces;
    }

    struct Outcome
    {
        std::vector<size_t> h1_expected_per_level;
        size_t h1_faces = 0;
        std::vector<std::string> problems;
    };

    // Three levels on [0, 2] x [0, 1], degree (2, 2): s halved per level, t
    // never refined.  Leaves are (level, patch, s element).
    Outcome threeLevels( const std::vector<std::array<size_t, 3>>& requested )
    {
        const double ptol = 1e-10;
        const KnotVector t( { 0, 0, 0, 1, 1, 1 }, ptol );
        std::vector<std::shared_ptr<const MultiPatchSplineSpace>> levels;
        Fixture fx;
        for( size_t level = 0; level < 3; level++ )
        {
            const size_t n = size_t( 2 ) << level;
            std::vector<double> bp;
            for( size_t i = 0; i <= n; i++ ) bp.push_back( 2.0 * static_cast<double>( i ) / static_cast<double>( n ) );
            std::vector<double> knots{ 0, 0 };
            knots.insert( knots.end(), bp.begin(), bp.end() );
            knots.insert( knots.end(), { 2, 2 } );
            const auto patch = makePatch( { KnotVector( knots, ptol ), t }, { 2, 2 } );
            levels.push_back( std::make_shared<const MultiPatchSplineSpace>(
                buildH1MultiPatchSplineSpace( { patch, patch }, sMaxToSMin() ) ) );
            fx.breakpoints.push_back( { bp, std::vector<double>{ 0, 1 } } );
            fx.leaves.emplace_back();
        }
        for( const auto& [level, patch, s] : requested )
            fx.leaves.at( level ).push_back( patchTopCellAt( levelCMap( *levels.at( level ) ), patch, { s, 0 } ) );

        const HierarchicalMultiPatchSplineSpace h1 = buildHierarchicalSplineSpace( levels, fx.leaves );
        const VectorConformingHierarchicalMultiPatchSplineSpace hdiv = buildHDivHierarchicalMultiPatchSplineSpace( h1 );
        const VectorConformingHierarchicalMultiPatchSplineSpace hcurl =
            buildHCurlHierarchicalMultiPatchSplineSpace( h1 );

        Outcome out;
        const auto activity = [&]( const HierarchyView& v ) {
            INFO( v.name );
            REQUIRE( v.levels.size() == 3 );
            const std::set<LevelFunction> expected = expectedActive( v, fx, requestedLeaves( v, fx ) );
            checkActivity( v, expected, out.problems );
            return expected;
        };
        const HierarchyView h1_view = viewOf( "H1", h1 );
        out.h1_expected_per_level.assign( 3, 0 );
        for( const auto& [level, f] : activity( h1_view ) ) out.h1_expected_per_level.at( level )++;
        out.h1_faces = checkH1Conformity( h1_view, fx, { 2, 2 }, out.problems );
        activity( viewOf( "H(div)", hdiv ) );
        activity( viewOf( "H(curl)", hcurl ) );
        return out;
    }
}

TEST_CASE( "Three levels, jump 2 at the interface only: THB activity and H1 conformity",
           "[shared_interface_refinement]" )
{
    // Patch 0: level 1 on [0, 1.5], level 2 on [1.5, 2].  Patch 1: level 0.
    const Outcome o =
        threeLevels( { { 0, 1, 0 }, { 0, 1, 1 }, { 1, 0, 0 }, { 1, 0, 1 }, { 1, 0, 2 }, { 2, 0, 6 }, { 2, 0, 7 } } );
    CHECK( o.h1_expected_per_level == std::vector<size_t>{ 12, 15, 3 } );
    CHECK( o.h1_faces == 5 );
    INFO( joined( o.problems ) );
    CHECK( o.problems.empty() );
}

TEST_CASE( "Three levels, jump 2 inside a patch: THB activity and H1 conformity", "[shared_interface_refinement]" )
{
    // Patch 0: level 0 on [0, 1], level 2 on [1, 1.5], level 1 on [1.5, 2].
    // Patch 1: level 0.  Jump 2 at s = 1 inside patch 0.
    const Outcome o = threeLevels( { { 0, 0, 0 }, { 0, 1, 0 }, { 0, 1, 1 }, { 1, 0, 3 }, { 2, 0, 4 }, { 2, 0, 5 } } );
    CHECK( o.h1_expected_per_level == std::vector<size_t>{ 21, 3, 0 } );
    CHECK( o.h1_faces == 4 );
    INFO( joined( o.problems ) );
    CHECK( o.problems.empty() );
}

// Step 6: patch-interface enumeration under asymmetric leaves (E3).  Both
// patches share the breakpoints of each level; patch 0's s-max side is glued
// to patch 1's s-min side with t running the same way on both.  A reversed
// layout glues s-max to s-max with t reversed, and mirrors patch 1's t
// breakpoints.
namespace
{
    using ElementKey = std::array<size_t, 4>; // level, patch, s element, t element

    struct TwoPatchLayout
    {
        std::string name;
        std::vector<std::array<std::vector<double>, 2>> breakpoints; // per level: s, t of patch 0
        std::vector<ElementKey> leaves;
        bool reversed = false;
    };

    InternalConnectionsMap sMaxToSMax()
    {
        const ConstituentSide first{ 0, ElementSide( 0, false ).sideId() };
        const ConstituentSide second{ 1, ElementSide( 0, false ).sideId() };
        return { { first, { TPPermutation::Flip1d, second } }, { second, { TPPermutation::Flip1d, first } } };
    }

    std::array<std::vector<double>, 2>
        patchBreakpoints( const TwoPatchLayout& layout, const size_t level, const size_t patch )
    {
        std::array<std::vector<double>, 2> bp = layout.breakpoints.at( level );
        if( layout.reversed and patch == 1 )
        {
            const double ends = bp[1].front() + bp[1].back();
            std::reverse( bp[1].begin(), bp[1].end() );
            for( double& t : bp[1] ) t = ends - t;
        }
        return bp;
    }

    std::string describe( const ElementKey& k )
    {
        std::ostringstream out;
        out << "p" << k[1] << " L" << k[0] << " (" << k[2] << "," << k[3] << ")";
        return out.str();
    }

    // Throws unless the element is named by a leaf dart.
    ElementKey keyOf( const HierarchicalMultiPatchCombinatorialMap& cmap, const Cell& element )
    {
        const auto [level, level_d] = cmap.unrefinedAncestorDartOfCell( element );
        const MultiPatchCombinatorialMap& level_cmap = *cmap.refinementLevels().at( level );
        const auto [patch, patch_d] = level_cmap.toLocalDart( level_d );
        const FullyUnflattenedDart unflat = unflattenFull( *level_cmap.constituents().at( patch ), patch_d );
        return { level, patch, unflat.unflat_darts.at( 0 ).id(), unflat.unflat_darts.at( 1 ).id() };
    }

    Box boxOf( const TwoPatchLayout& layout, const ElementKey& k )
    {
        const auto bp = patchBreakpoints( layout, k[0], k[1] );
        return { std::array<double, 2>{ bp[0].at( k[2] ), bp[0].at( k[2] + 1 ) },
                 std::array<double, 2>{ bp[1].at( k[3] ), bp[1].at( k[3] + 1 ) } };
    }

    bool onInterface( const TwoPatchLayout& layout, const ElementKey& k )
    {
        const std::vector<double>& s = layout.breakpoints.front()[0];
        const bool s_max = k[1] == 0 or layout.reversed;
        return std::abs( boxOf( layout, k )[0][s_max ? 1 : 0] - ( s_max ? s.back() : s.front() ) ) < geom_tol;
    }

    // A patch 1 t in patch 0's t.
    double patchZeroT( const TwoPatchLayout& layout, const size_t patch, const double t )
    {
        const std::vector<double>& bp = layout.breakpoints.front()[1];
        return layout.reversed and patch == 1 ? bp.front() + bp.back() - t : t;
    }

    // The t interval of a leaf's interface side, in patch 0's t.
    std::array<double, 2> interfaceInterval( const TwoPatchLayout& layout, const ElementKey& k )
    {
        const std::array<double, 2> t = boxOf( layout, k )[1];
        const double a = patchZeroT( layout, k[1], t[0] );
        const double b = patchZeroT( layout, k[1], t[1] );
        return { std::min( a, b ), std::max( a, b ) };
    }

    // Leaf pairs, (patch 0, patch 1), whose interface sides overlap in t.
    std::set<std::pair<ElementKey, ElementKey>> expectedInterfacePairs( const TwoPatchLayout& layout )
    {
        std::set<std::pair<ElementKey, ElementKey>> out;
        for( const ElementKey& a : layout.leaves )
            for( const ElementKey& b : layout.leaves )
            {
                if( a[1] != 0 or b[1] != 1 or not onInterface( layout, a ) or not onInterface( layout, b ) ) continue;
                const std::array<double, 2> ta = interfaceInterval( layout, a );
                const std::array<double, 2> tb = interfaceInterval( layout, b );
                if( std::min( ta[1], tb[1] ) - std::max( ta[0], tb[0] ) > geom_tol ) out.insert( { a, b } );
            }
        return out;
    }

    // The side of box a that box b touches along a positive length.
    std::optional<ElementSide> facingSide( const Box& a, const Box& b )
    {
        for( size_t axis = 0; axis < 2; axis++ )
        {
            const size_t other = 1 - axis;
            if( std::min( a[other][1], b[other][1] ) - std::max( a[other][0], b[other][0] ) <= geom_tol ) continue;
            if( std::abs( a[axis][1] - b[axis][0] ) < geom_tol ) return ElementSide( axis, false );
            if( std::abs( a[axis][0] - b[axis][1] ) < geom_tol ) return ElementSide( axis, true );
        }
        return std::nullopt;
    }

    // Same-level leaf pairs of one patch whose boxes share an edge, smaller key first.
    std::set<std::pair<ElementKey, ElementKey>> expectedInteriorPairs( const TwoPatchLayout& layout )
    {
        std::set<std::pair<ElementKey, ElementKey>> out;
        for( const ElementKey& a : layout.leaves )
            for( const ElementKey& b : layout.leaves )
                if( a < b and a[0] == b[0] and a[1] == b[1] and
                    facingSide( boxOf( layout, a ), boxOf( layout, b ) ).has_value() )
                    out.insert( { a, b } );
        return out;
    }

    struct EnumerationOutcome
    {
        size_t expected_pairs = 0;
        std::vector<std::string> problems;
        std::vector<std::string> interior_problems;
    };

    EnumerationOutcome patchInterfaceEnumeration( const TwoPatchLayout& layout )
    {
        const double ptol = 1e-10;
        std::vector<std::shared_ptr<const MultiPatchSplineSpace>> levels;
        std::vector<std::vector<Cell>> leaves( layout.breakpoints.size() );
        for( size_t level = 0; level < layout.breakpoints.size(); level++ )
        {
            std::vector<std::shared_ptr<const TPSplineSpace>> patches;
            for( size_t patch = 0; patch < 2; patch++ )
            {
                SmallVector<KnotVector, 3> kvs;
                for( const std::vector<double>& dir : patchBreakpoints( layout, level, patch ) )
                {
                    std::vector<double> knots{ dir.front(), dir.front() };
                    knots.insert( knots.end(), dir.begin(), dir.end() );
                    knots.insert( knots.end(), { dir.back(), dir.back() } );
                    kvs.push_back( KnotVector( knots, ptol ) );
                }
                patches.push_back( makePatch( kvs, { 2, 2 } ) );
            }
            levels.push_back( std::make_shared<const MultiPatchSplineSpace>(
                buildH1MultiPatchSplineSpace( patches, layout.reversed ? sMaxToSMax() : sMaxToSMin() ) ) );
        }
        for( const auto& [level, patch, s, t] : layout.leaves )
            leaves.at( level ).push_back( patchTopCellAt( levelCMap( *levels.at( level ) ), patch, { s, t } ) );

        const HierarchicalMultiPatchSplineSpace h1 = buildHierarchicalSplineSpace( levels, leaves );
        const VectorConformingHierarchicalMultiPatchSplineSpace hdiv = buildHDivHierarchicalMultiPatchSplineSpace( h1 );
        const VectorConformingHierarchicalMultiPatchSplineSpace hcurl =
            buildHCurlHierarchicalMultiPatchSplineSpace( h1 );

        const std::set<ElementKey> leaf_keys( layout.leaves.begin(), layout.leaves.end() );
        const std::set<std::pair<ElementKey, ElementKey>> expected = expectedInterfacePairs( layout );
        EnumerationOutcome out{ expected.size(), {}, {} };
        const auto check = [&]( const std::string& name, const SplineSpace& ss ) {
            const auto& cmap = dynamic_cast<const HierarchicalMultiPatchCombinatorialMap&>(
                ss.basisComplex().parametricAtlas().cmap() );
            const std::string prefix = layout.name + ", " + name + ": ";
            std::multiset<std::pair<ElementKey, ElementKey>> actual;
            std::map<ElementKey, double> covered;
            for( const TraceMeshInterface& iface : patchTraceMeshInterfaces( ss ) )
            {
                if( iface.type != TraceMeshEntityType::PatchInterface or not iface.second.has_value() )
                {
                    out.problems.push_back( prefix + "a record is not a two-sided patch interface" );
                    continue;
                }
                ElementKey first;
                ElementKey second;
                try
                {
                    first = keyOf( cmap, iface.first.element );
                    second = keyOf( cmap, iface.second->element );
                }
                catch( const std::exception& e )
                {
                    out.problems.push_back( prefix + "a patch interface side is not named by a leaf dart: " +
                                            e.what() );
                    continue;
                }
                if( first[0] != iface.first.level or second[0] != iface.second->level or
                    first[1] != iface.first.patch_id or second[1] != iface.second->patch_id or first[1] == second[1] )
                    out.problems.push_back( prefix + "record fields disagree with " + describe( first ) + " / " +
                                            describe( second ) );
                for( const ElementKey& k : { first, second } )
                    if( not leaf_keys.contains( k ) ) out.problems.push_back( prefix + "non-leaf " + describe( k ) );
                try
                {
                    traceInterfaceElement( ss, iface );
                }
                catch( const std::exception& e )
                {
                    out.problems.push_back( prefix + "traceInterfaceElement on " + describe( first ) + " / " +
                                            describe( second ) + " throws: " + e.what() );
                }
                actual.insert( first[1] == 0 ? std::pair{ first, second } : std::pair{ second, first } );

                // Both sides place each fragment point at one t of the intersection
                // of the two leaf sides; the finer side is whole.
                const std::array<double, 2> ta = interfaceInterval( layout, first );
                const std::array<double, 2> tb = interfaceInterval( layout, second );
                const std::array<double, 2> fragment = { std::max( ta[0], tb[0] ), std::min( ta[1], tb[1] ) };
                for( const auto& [k, side] : { std::pair{ first, &iface.first }, std::pair{ second, &*iface.second } } )
                {
                    const TraceSideEmbedding& e = side->embedding;
                    if( e.offset.size() != 1 )
                    {
                        out.problems.push_back( prefix + describe( k ) + " has an embedding of rank " +
                                                std::to_string( e.offset.size() ) );
                        continue;
                    }
                    covered[k] += e.scale.at( 0 );
                    if( k[0] == std::max( first[0], second[0] ) and
                        ( std::abs( e.offset.at( 0 ) ) > geom_tol or std::abs( e.scale.at( 0 ) - 1.0 ) > geom_tol ) )
                        out.problems.push_back( prefix + "finer side " + describe( k ) + " is not whole" );
                    const Box b = boxOf( layout, k );
                    for( const double xi : { 0.0, 0.25, 1.0 } )
                    {
                        const double y = embedTraceFragmentPoint( e, Eigen::VectorXd::Constant( 1, xi ) )( 0 );
                        const double t = patchZeroT( layout, k[1], b[1][0] + y * ( b[1][1] - b[1][0] ) );
                        const double expected = fragment[0] + xi * ( fragment[1] - fragment[0] );
                        if( std::abs( t - expected ) > geom_tol )
                        {
                            std::ostringstream text;
                            text << prefix << describe( first ) << " / " << describe( second ) << ": "
                                 << describe( k ) << " places fragment point " << xi << " at t = " << t
                                 << ", not " << expected;
                            out.problems.push_back( text.str() );
                        }
                    }
                }
            }
            for( const std::vector<TraceMeshInterface>& records :
                 { boundaryTraceMeshInterfaces( ss ), interiorTraceMeshInterfaces( ss ) } )
                for( const TraceMeshInterface& iface : records )
                    for( const TraceMeshSide* side : { &iface.first, iface.second ? &*iface.second : nullptr } )
                    {
                        if( side == nullptr ) continue;
                        const std::string kind = iface.isBoundary() ? "boundary" : "interior";
                        try
                        {
                            const ElementKey k = keyOf( cmap, side->element );
                            if( not leaf_keys.contains( k ) or k[0] != side->level or k[1] != side->patch_id )
                                out.problems.push_back( prefix + kind + " record names " + describe( k ) );
                        }
                        catch( const std::exception& e )
                        {
                            out.problems.push_back( prefix + kind + " record side " +
                                                    std::to_string( side->side.sideId() ) + " in patch " +
                                                    std::to_string( side->patch_id ) + " at level " +
                                                    std::to_string( side->level ) +
                                                    " is not named by a leaf dart: " + e.what() );
                        }
                    }

            const std::set<std::pair<ElementKey, ElementKey>> expected_interior = expectedInteriorPairs( layout );
            std::multiset<std::pair<ElementKey, ElementKey>> interior;
            for( const TraceMeshInterface& iface : interiorTraceMeshInterfaces( ss ) )
            {
                ElementKey first;
                ElementKey second;
                try
                {
                    first = keyOf( cmap, iface.first.element );
                    second = keyOf( cmap, iface.second.value().element );
                }
                catch( const std::exception& )
                {
                    continue; // reported by the leaf-dart gate above
                }
                const std::string text = describe( first ) + " side " + std::to_string( iface.first.side.sideId() ) +
                                         " / " + describe( second ) + " side " +
                                         std::to_string( iface.second->side.sideId() );
                if( facingSide( boxOf( layout, first ), boxOf( layout, second ) ) != iface.first.side or
                    facingSide( boxOf( layout, second ), boxOf( layout, first ) ) != iface.second->side )
                    out.interior_problems.push_back( prefix + "interior sides do not face each other: " + text );
                if( iface.first.side.lower )
                    out.interior_problems.push_back( prefix + "interior first side is a min side: " + text );
                interior.insert( std::minmax( first, second ) );
            }
            for( const auto& pair : interior )
            {
                const std::string text = describe( pair.first ) + " / " + describe( pair.second );
                if( interior.count( pair ) > 1 )
                    out.interior_problems.push_back( prefix + "duplicate interior " + text );
                if( not expected_interior.contains( pair ) )
                    out.interior_problems.push_back( prefix + "unexpected interior " + text );
            }
            for( const auto& pair : expected_interior )
                if( not interior.contains( pair ) )
                    out.interior_problems.push_back( prefix + "missing interior " + describe( pair.first ) + " / " +
                                                     describe( pair.second ) );

            for( const ElementKey& k : layout.leaves )
                if( onInterface( layout, k ) and std::abs( covered[k] - 1.0 ) > geom_tol )
                    out.problems.push_back( prefix + "fragments cover " + std::to_string( covered[k] ) + " of " +
                                            describe( k ) + "'s side" );
            for( const auto& pair : actual )
            {
                const std::string text = describe( pair.first ) + " / " + describe( pair.second );
                if( actual.count( pair ) > 1 ) out.problems.push_back( prefix + "duplicate " + text );
                if( not expected.contains( pair ) ) out.problems.push_back( prefix + "unexpected " + text );
            }
            for( const auto& pair : expected )
                if( not actual.contains( pair ) )
                    out.problems.push_back( prefix + "missing " + describe( pair.first ) + " / " +
                                            describe( pair.second ) );
        };
        check( "H1", h1 );
        check( "H(div)", hdiv );
        check( "H(curl)", hcurl );
        return out;
    }

    std::vector<TwoPatchLayout> interfaceLayouts()
    {
        using BP = std::array<std::vector<double>, 2>;
        const auto bp = []( const std::vector<double>& s, const std::vector<double>& t ) { return BP{ s, t }; };
        const std::vector<BP> normal = { bp( { 0, 1, 2 }, { 0, 1 } ), bp( { 0, 0.5, 1, 1.5, 2 }, { 0, 1 } ) };
        const std::vector<BP> tangential = { bp( { 0, 1 }, { 0, 1 } ), bp( { 0, 0.5, 1 }, { 0, 0.5, 1 } ) };
        const std::vector<BP> non_dyadic = { bp( { 0, 1 }, { 0, 1 } ), bp( { 0, 0.5, 1 }, { 0, 1.0 / 3.0, 1 } ) };
        const std::vector<BP> three = { bp( { 0, 1 }, { 0, 1 } ),
                                        bp( { 0, 0.5, 1 }, { 0, 0.5, 1 } ),
                                        bp( { 0, 0.25, 0.5, 0.75, 1 }, { 0, 0.25, 0.5, 0.75, 1 } ) };
        const std::vector<ElementKey> fine_p0 = {
            { 0, 1, 0, 0 }, { 1, 0, 0, 0 }, { 1, 0, 1, 0 }, { 1, 0, 0, 1 }, { 1, 0, 1, 1 } };
        const std::vector<ElementKey> fine_p1 = {
            { 0, 0, 0, 0 }, { 1, 1, 0, 0 }, { 1, 1, 1, 0 }, { 1, 1, 0, 1 }, { 1, 1, 1, 1 } };
        return {
            { "normal, patch 0 refined",
              normal,
              { { 0, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 1, 1, 0 }, { 1, 0, 2, 0 }, { 1, 0, 3, 0 } } },
            { "normal, patch 1 refined",
              normal,
              { { 0, 0, 0, 0 }, { 0, 0, 1, 0 }, { 0, 1, 1, 0 }, { 1, 1, 0, 0 }, { 1, 1, 1, 0 } } },
            { "tangential, patch 0 refined", tangential, fine_p0 },
            { "tangential, patch 1 refined", tangential, fine_p1 },
            { "non-dyadic, patch 0 refined", non_dyadic, fine_p0 },
            { "non-dyadic, patch 1 refined", non_dyadic, fine_p1 },
            { "three levels, patch 0 refined",
              three,
              { { 0, 1, 0, 0 },
                { 1, 0, 0, 0 },
                { 1, 0, 1, 0 },
                { 1, 0, 0, 1 },
                { 2, 0, 2, 2 },
                { 2, 0, 3, 2 },
                { 2, 0, 2, 3 },
                { 2, 0, 3, 3 } } },
            { "three levels, patch 1 refined",
              three,
              { { 0, 0, 0, 0 },
                { 1, 1, 0, 0 },
                { 1, 1, 1, 0 },
                { 1, 1, 1, 1 },
                { 2, 1, 0, 2 },
                { 2, 1, 1, 2 },
                { 2, 1, 0, 3 },
                { 2, 1, 1, 3 } } },
            { "three levels, crossed",
              three,
              { { 1, 0, 0, 0 },
                { 1, 0, 0, 1 },
                { 1, 0, 1, 1 },
                { 2, 0, 2, 0 },
                { 2, 0, 3, 0 },
                { 2, 0, 2, 1 },
                { 2, 0, 3, 1 },
                { 1, 1, 0, 0 },
                { 1, 1, 1, 0 },
                { 1, 1, 1, 1 },
                { 2, 1, 0, 2 },
                { 2, 1, 1, 2 },
                { 2, 1, 0, 3 },
                { 2, 1, 1, 3 } } },
            { "reversed non-dyadic, patch 0 refined", non_dyadic, fine_p0, true },
            { "reversed non-dyadic, patch 1 refined", non_dyadic, fine_p1, true },
        };
    }
}

TEST_CASE( "Patch-interface enumeration under asymmetric leaves: preconditions", "[shared_interface_refinement]" )
{
    // Hand-counted leaf pairs across the interface and same-level leaf pairs
    // within a patch; the requested leaves partition each patch.
    const std::vector<size_t> pairs = { 1, 1, 2, 2, 2, 2, 3, 3, 4, 2, 2 };
    const std::vector<size_t> interior_pairs = { 2, 2, 4, 4, 4, 4, 6, 6, 12, 4, 4 };
    const std::vector<TwoPatchLayout> layouts = interfaceLayouts();
    REQUIRE( layouts.size() == pairs.size() );
    for( size_t i = 0; i < layouts.size(); i++ )
    {
        const TwoPatchLayout& layout = layouts.at( i );
        INFO( layout.name );
        for( size_t patch = 0; patch < 2; patch++ )
        {
            double area = 0.0;
            for( const ElementKey& a : layout.leaves )
            {
                if( a[1] != patch ) continue;
                area += measure( boxOf( layout, a ) );
                for( const ElementKey& b : layout.leaves )
                    if( b[1] == patch and a != b )
                        CHECK( overlap( boxOf( layout, a ), boxOf( layout, b ) ) < geom_tol );
            }
            const auto& bp = layout.breakpoints.front();
            CHECK( std::abs( area - ( bp[0].back() - bp[0].front() ) * ( bp[1].back() - bp[1].front() ) ) < geom_tol );
        }
        CHECK( patchInterfaceEnumeration( layout ).expected_pairs == pairs.at( i ) );
        CHECK( expectedInteriorPairs( layout ).size() == interior_pairs.at( i ) );
    }
}

TEST_CASE( "Patch-interface enumeration under asymmetric leaves: each leaf pair once, on their common fragment",
           "[shared_interface_refinement]" )
{
    std::vector<std::string> problems;
    for( const TwoPatchLayout& layout : interfaceLayouts() )
    {
        const EnumerationOutcome o = patchInterfaceEnumeration( layout );
        problems.insert( problems.end(), o.problems.begin(), o.problems.end() );
    }
    INFO( joined( problems ) );
    CHECK( problems.empty() );
}

// outstanding_issues.txt item 18.  Coarse-fine faces within a patch are not
// enumerated (item 7).
TEST_CASE( "Interior enumeration under asymmetric leaves: every same-level leaf pair once, on facing sides",
           "[shared_interface_refinement]" )
{
    std::vector<std::string> problems;
    for( const TwoPatchLayout& layout : interfaceLayouts() )
    {
        const EnumerationOutcome o = patchInterfaceEnumeration( layout );
        problems.insert( problems.end(), o.interior_problems.begin(), o.interior_problems.end() );
    }
    INFO( joined( problems ) );
    CHECK( problems.empty() );
}

// Step 6: a level whose breakpoints contradict the integer-ratio child
// relation.  The coarse s elements [0, 0.5] and [0.5, 1] split into two and
// four fine elements; the ratio is 3.
namespace
{
    KnotVector openKnots( const std::vector<double>& bp )
    {
        std::vector<double> knots{ bp.front(), bp.front() };
        knots.insert( knots.end(), bp.begin(), bp.end() );
        knots.insert( knots.end(), { bp.back(), bp.back() } );
        return KnotVector( knots, 1e-10 );
    }

    std::string builtSpace( const SplineSpace& ss )
    {
        const param::ParametricAtlas& atlas = ss.basisComplex().parametricAtlas();
        double area = 0.0;
        for( const Cell& c : topCells( atlas.cmap() ) ) area += atlas.parametricLengths( c ).prod();
        return "built, leaves cover area " + std::to_string( area ) + ", " + std::to_string( ss.numFunctions() ) +
               " functions";
    }
}

TEST_CASE( "Uneven child counts are rejected when a hierarchy is built", "[shared_interface_refinement]" )
{
    const auto coarse = makePatch( { openKnots( { 0, 0.5, 1 } ), openKnots( { 0, 1 } ) }, { 2, 2 } );
    const auto fine = makePatch( { openKnots( { 0, 0.25, 0.5, 0.6, 0.7, 0.8, 1 } ), openKnots( { 0, 1 } ) }, { 2, 2 } );
    const auto tpCell = []( const TPSplineSpace& ss, const size_t s ) {
        const auto& cmap = dynamic_cast<const TPCombinatorialMap&>( ss.basisComplex().parametricAtlas().cmap() );
        return Cell( flattenFull( cmap, FullyUnflattenedDart( { Dart( s ), Dart( 0 ) } ) ), 2 );
    };
    // Coarse-only leaves, and fine leaves at the index children of coarse 0,
    // which reach s = 0.6 and so overlap coarse 1.
    const std::vector<std::vector<Cell>> tp_coarse = { { tpCell( *coarse, 0 ), tpCell( *coarse, 1 ) }, {} };
    const std::vector<std::vector<Cell>> tp_mixed = { { tpCell( *coarse, 1 ) },
                                                      { tpCell( *fine, 0 ), tpCell( *fine, 1 ), tpCell( *fine, 2 ) } };

    std::vector<std::shared_ptr<const MultiPatchSplineSpace>> mp_levels;
    std::vector<std::shared_ptr<const VectorConformingMultiPatchSplineSpace>> hdiv_levels;
    for( const auto& patch : { coarse, fine } )
    {
        mp_levels.push_back( std::make_shared<const MultiPatchSplineSpace>(
            buildH1MultiPatchSplineSpace( { patch, patch }, sMaxToSMin() ) ) );
        hdiv_levels.push_back( std::make_shared<const VectorConformingMultiPatchSplineSpace>(
            buildHDivMultiPatchSplineSpace( *mp_levels.back() ) ) );
    }
    const auto mpCell = [&]( const size_t level, const size_t patch, const size_t s ) {
        return patchTopCellAt( levelCMap( *mp_levels.at( level ) ), patch, { s, 0 } );
    };
    const std::vector<std::vector<Cell>> mp_mixed = { { mpCell( 0, 0, 1 ), mpCell( 0, 1, 0 ), mpCell( 0, 1, 1 ) },
                                                      { mpCell( 1, 0, 0 ), mpCell( 1, 0, 1 ), mpCell( 1, 0, 2 ) } };

    std::vector<std::string> problems;
    const auto attempt = [&]( const std::string& name, const std::function<std::string()>& build ) {
        try
        {
            problems.push_back( name + ": " + build() );
        }
        catch( const std::invalid_argument& e )
        {
            if( not std::string( e.what() ).contains( "must split into the same number of children" ) )
                problems.push_back( name + ": throws another invalid_argument: " + e.what() );
        }
        catch( const std::exception& e )
        {
            problems.push_back( name + ": throws other than invalid_argument: " + e.what() );
        }
    };
    const std::vector<std::shared_ptr<const TPSplineSpace>> tp_levels = { coarse, fine };
    attempt( "single patch, coarse leaves",
             [&] { return builtSpace( buildHierarchicalSplineSpace( tp_levels, tp_coarse ) ); } );
    attempt( "single patch, mixed leaves",
             [&] { return builtSpace( buildHierarchicalSplineSpace( tp_levels, tp_mixed ) ); } );
    attempt( "two patches H1, mixed leaves",
             [&] { return builtSpace( buildHierarchicalSplineSpace( mp_levels, mp_mixed ) ); } );
    attempt( "two patches H(div), mixed leaves",
             [&] { return builtSpace( buildHierarchicalSplineSpace( hdiv_levels, mp_mixed ) ); } );
    INFO( joined( problems ) );
    CHECK( problems.empty() );
}

// Item 1 step 7: physical trace conformity on the step-6 fragments.  The
// geometry is a level-0 spline whose control points map each patch's
// Greville points smoothly.  Both sides must reach one physical point, and
// full-vector-jump observations show that full vector continuity is not imposed.
namespace
{
    enum class TraceKind
    {
        H1,
        HDiv,
        HCurl
    };

    struct LayoutHierarchy
    {
        std::vector<std::shared_ptr<const TPSplineSpace>> level_zero_patches;
        std::vector<std::shared_ptr<const MultiPatchSplineSpace>> levels;
        std::vector<std::vector<Cell>> leaves;
    };

    // The hierarchy patchInterfaceEnumeration builds.
    LayoutHierarchy layoutHierarchy( const TwoPatchLayout& layout )
    {
        const double ptol = 1e-10;
        LayoutHierarchy out;
        out.leaves.resize( layout.breakpoints.size() );
        for( size_t level = 0; level < layout.breakpoints.size(); level++ )
        {
            std::vector<std::shared_ptr<const TPSplineSpace>> patches;
            for( size_t patch = 0; patch < 2; patch++ )
            {
                SmallVector<KnotVector, 3> kvs;
                for( const std::vector<double>& dir : patchBreakpoints( layout, level, patch ) )
                {
                    std::vector<double> knots{ dir.front(), dir.front() };
                    knots.insert( knots.end(), dir.begin(), dir.end() );
                    knots.insert( knots.end(), { dir.back(), dir.back() } );
                    kvs.push_back( KnotVector( knots, ptol ) );
                }
                patches.push_back( makePatch( kvs, { 2, 2 } ) );
            }
            if( level == 0 ) out.level_zero_patches = patches;
            out.levels.push_back( std::make_shared<const MultiPatchSplineSpace>(
                buildH1MultiPatchSplineSpace( patches, layout.reversed ? sMaxToSMax() : sMaxToSMin() ) ) );
        }
        for( const auto& [level, patch, s, t] : layout.leaves )
            out.leaves.at( level ).push_back( patchTopCellAt( levelCMap( *out.levels.at( level ) ), patch, { s, t } ) );
        return out;
    }

    // Patch 1 placed so that its glued side meets patch 0's s-max side.
    Eigen::VectorXd layoutGlobalPoint( const TwoPatchLayout& layout, const size_t patch, const Eigen::VectorXd& p )
    {
        if( patch == 0 ) return p;
        const std::vector<double>& s = layout.breakpoints.front()[0];
        Eigen::VectorXd out = p;
        out( 0 ) = layout.reversed ? 2.0 * s.back() - p( 0 ) : s.back() + p( 0 ) - s.front();
        out( 1 ) = patchZeroT( layout, patch, p( 1 ) );
        return out;
    }

    // A curved control net; positive orientation is checked at every evaluated point.
    Eigen::VectorXd curvedMap( const Eigen::VectorXd& p )
    {
        Eigen::VectorXd out = p;
        out( 0 ) += 0.2 * p( 1 ) * p( 1 );
        out( 1 ) += 0.1 * p( 0 ) * p( 0 );
        if( p.size() == 3 )
        {
            out( 0 ) += 0.1 * p( 2 ) * p( 2 );
            out( 2 ) += 0.15 * p( 0 ) * p( 1 );
        }
        return out;
    }

    struct SplineGeometry
    {
        std::shared_ptr<const MultiPatchSplineSpace> ss;
        Eigen::MatrixXd cpts; // coordinate by function
    };

    // Throws unless the patches give each shared function one control point.
    SplineGeometry splineGeometry( const std::shared_ptr<const MultiPatchSplineSpace>& ss,
                                   const std::vector<std::shared_ptr<const TPSplineSpace>>& patches,
                                   const std::function<Eigen::VectorXd( size_t, const Eigen::VectorXd& )>& to_global )
    {
        std::vector<Eigen::MatrixXd> patch_cpts;
        for( size_t patch = 0; patch < patches.size(); patch++ )
        {
            const Eigen::MatrixXd greville = grevillePoints( *patches.at( patch ) );
            Eigen::MatrixXd cpts( greville.rows(), greville.cols() );
            for( Eigen::Index i = 0; i < greville.rows(); i++ )
                cpts.row( i ) = curvedMap( to_global( patch, greville.row( i ).transpose() ) ).transpose();
            patch_cpts.push_back( cpts );
        }
        const Eigen::MatrixXd cpts = multiPatchCoefficients( *ss, patch_cpts );
        const std::vector<Eigen::MatrixXd> split = splitMultiPatchCoefficients( *ss, cpts );
        for( size_t patch = 0; patch < patches.size(); patch++ )
            if( ( split.at( patch ) - patch_cpts.at( patch ) ).cwiseAbs().maxCoeff() > geom_tol )
                throw std::runtime_error( "patch control points disagree on a shared function" );
        return { ss, cpts.transpose() };
    }

    // The level-0 element of a patch containing a patch-parametric point, and
    // the point's parent coordinates there; empty unless exactly one.
    std::optional<std::pair<Cell, Eigen::VectorXd>>
        levelZeroLocation( const MultiPatchSplineSpace& ss, const size_t patch, const Eigen::VectorXd& point )
    {
        const auto& atlas = ss.basisComplex().parametricAtlas();
        const Eigen::Index dim = point.size();
        std::optional<std::pair<Cell, Eigen::VectorXd>> out;
        for( const Cell& c : topCellsForPatch( levelCMap( ss ), patch ) )
        {
            const Eigen::VectorXd xi = ( point - atlas.parametricStarts( c ).head( dim ) )
                                           .cwiseQuotient( atlas.parametricLengths( c ).head( dim ) );
            if( ( xi.array() < -geom_tol ).any() or ( xi.array() > 1.0 + geom_tol ).any() ) continue;
            if( out.has_value() ) return std::nullopt;
            out.emplace( c, xi );
        }
        return out;
    }

    // The evaluator reads no zero flags.
    param::ParentPoint cubePoint( const Eigen::VectorXd& xi )
    {
        return param::ParentPoint(
            param::cubeDomain( xi.size() ), Vector3dMax( xi ), param::BaryCoordIsZeroVec( 2 * xi.size(), false ) );
    }

    // n x v, a scalar in 2D.
    Eigen::VectorXd crossWithNormal( const Eigen::VectorXd& n, const Eigen::VectorXd& v )
    {
        if( n.size() == 2 ) return Eigen::VectorXd::Constant( 1, n( 0 ) * v( 1 ) - n( 1 ) * v( 0 ) );
        return Eigen::Vector3d( n ).cross( Eigen::Vector3d( v ) );
    }

    struct SideTrace
    {
        Eigen::VectorXd x;
        Eigen::VectorXd normal; // outward unit normal
        std::map<FunctionId, Eigen::VectorXd> trace;
        std::map<FunctionId, Eigen::VectorXd> spatial;
    };

    // Evaluates at fragment point xi, placed on the side's leaf by its embedding.
    SideTrace sideTrace( const SplineSpace& ss,
                         const TraceKind kind,
                         const TraceMeshSide& side,
                         const Eigen::VectorXd& xi,
                         const SplineGeometry& geom )
    {
        const size_t dim = ss.basisComplex().parametricAtlas().cmap().dim();
        const Eigen::VectorXd y = embedTraceFragmentPoint( side.embedding, xi );
        const std::vector<size_t> axes = tangentialAxes( dim, side.side.sideId() );
        Eigen::VectorXd leaf_xi( dim );
        leaf_xi( side.side.axis ) = side.side.lower ? 0.0 : 1.0;
        for( size_t k = 0; k < axes.size(); k++ ) leaf_xi( axes.at( k ) ) = y( k );

        eval::SplineSpaceEvaluator evals( ss, 0 );
        evals.localizeElement( side.element );
        evals.localizeParentPoint( cubePoint( leaf_xi ) );
        const auto located = levelZeroLocation( *geom.ss, side.patch_id, evals.evaluateParametricPoint() );
        if( not located.has_value() ) throw std::runtime_error( "the point is not in exactly one level-0 element" );
        eval::SplineSpaceEvaluator geom_evals( *geom.ss, 1 );
        geom_evals.localizeElement( located->first );
        geom_evals.localizeParentPoint( cubePoint( located->second ) );

        SideTrace out;
        out.x = geom_evals.evaluateManifold( geom.cpts );
        const Eigen::MatrixXd jacobian = geom_evals.evaluateParametricToSpatialJacobian( geom.cpts );
        if( jacobian.determinant() <= 0.0 ) throw std::runtime_error( "the geometry is inverted" );
        out.normal = jacobian.inverse().transpose().col( side.side.axis ) * ( side.side.lower ? -1.0 : 1.0 );
        out.normal.normalize();
        const Eigen::MatrixXd values =
            kind == TraceKind::H1     ? evals.evaluateBasisValuesAtParentPoint()
            : kind == TraceKind::HDiv ? eval::evaluateSpatialHDivBasisValues( evals, geom_evals, geom.cpts )
                                      : eval::evaluateSpatialHCurlBasisValues( evals, geom_evals, geom.cpts );
        const std::vector<FunctionId> conn = ss.connectivity( side.element );
        for( size_t i = 0; i < conn.size(); i++ )
        {
            const Eigen::VectorXd v = values.row( i ).transpose();
            out.spatial[conn.at( i )] = v;
            out.trace[conn.at( i )] = kind == TraceKind::H1     ? v
                                      : kind == TraceKind::HDiv ? Eigen::VectorXd::Constant( 1, v.dot( out.normal ) )
                                                                : crossWithNormal( out.normal, v );
        }
        return out;
    }

    bool traceClose( const Eigen::VectorXd& a, const Eigen::VectorXd& b )
    {
        return ( a - b ).norm() <= 1e-9 * std::max( { 1.0, a.norm(), b.norm() } );
    }

    std::string sideText( const TraceMeshSide& side )
    {
        return "p" + std::to_string( side.patch_id ) + " L" + std::to_string( side.level ) + " side " +
               std::to_string( side.side.sideId() );
    }

    struct ConformityOutcome
    {
        std::vector<std::string> problems;
        size_t records = 0;
        size_t comparisons = 0; // functions compared at a point
        // Observations are per (record, point, shared function), not unique functions.
        size_t nonzero_shared_trace_observations = 0; // trace above 1e-3
        size_t full_vector_jump_observations = 0;     // spatial value jumps
    };

    // H1 values agree.  With each side's outward normal, the H(div) normal
    // traces and the H(curl) n x u are opposite.  A function on one side only
    // has a zero trace.
    ConformityOutcome fragmentTraceConformity( const std::string& prefix,
                                               const SplineSpace& ss,
                                               const TraceKind kind,
                                               const SplineGeometry& geom,
                                               const std::vector<Eigen::VectorXd>& points )
    {
        ConformityOutcome out;
        const double sign = kind == TraceKind::H1 ? 1.0 : -1.0;
        for( const TraceMeshInterface& iface : patchTraceMeshInterfaces( ss ) )
        {
            if( not iface.second.has_value() )
            {
                out.problems.push_back( prefix + "a patch interface record has one side" );
                continue;
            }
            out.records++;
            for( const Eigen::VectorXd& xi : points )
            {
                std::ostringstream where;
                where << prefix << sideText( iface.first ) << " / " << sideText( *iface.second ) << " at "
                      << xi.transpose() << ": ";
                SideTrace a;
                SideTrace b;
                try
                {
                    a = sideTrace( ss, kind, iface.first, xi, geom );
                    b = sideTrace( ss, kind, *iface.second, xi, geom );
                }
                catch( const std::exception& e )
                {
                    out.problems.push_back( where.str() + e.what() );
                    continue;
                }
                if( ( a.x - b.x ).norm() > 1e-11 )
                {
                    std::ostringstream text;
                    text << where.str() << "physical points " << a.x.transpose() << " and " << b.x.transpose();
                    out.problems.push_back( text.str() );
                }
                if( ( a.normal + b.normal ).norm() > 1e-9 )
                    out.problems.push_back( where.str() + "outward normals are not opposite" );
                std::set<FunctionId> fids;
                for( const SideTrace* t : { &a, &b } )
                    for( const auto& [f, v] : t->trace ) fids.insert( f );
                bool nonzero = false;
                for( const FunctionId& f : fids )
                {
                    const bool in_a = a.trace.contains( f );
                    const bool in_b = b.trace.contains( f );
                    const Eigen::VectorXd zero = Eigen::VectorXd::Zero( ( in_a ? a : b ).trace.at( f ).size() );
                    const Eigen::VectorXd ta = in_a ? a.trace.at( f ) : zero;
                    const Eigen::VectorXd tb = sign * ( in_b ? b.trace.at( f ) : zero );
                    out.comparisons++;
                    if( not traceClose( ta, tb ) )
                    {
                        std::ostringstream text;
                        text << where.str() << "function " << f << " has traces " << ta.transpose() << " and "
                             << tb.transpose() << ( in_a and in_b ? "" : in_a ? " (first only)" : " (second only)" );
                        out.problems.push_back( text.str() );
                    }
                    if( not in_a or not in_b ) continue;
                    if( ta.norm() > 1e-3 )
                    {
                        out.nonzero_shared_trace_observations++;
                        nonzero = true;
                    }
                    if( kind != TraceKind::H1 and ( a.spatial.at( f ) - b.spatial.at( f ) ).norm() > 1e-6 )
                        out.full_vector_jump_observations++;
                }
                if( not nonzero ) out.problems.push_back( where.str() + "no shared function has a nonzero trace" );
            }
        }
        return out;
    }

    std::vector<std::pair<std::string, ConformityOutcome>> layoutTraceConformity( const TwoPatchLayout& layout )
    {
        const LayoutHierarchy hierarchy = layoutHierarchy( layout );
        const SplineGeometry geom =
            splineGeometry( hierarchy.levels.front(),
                            hierarchy.level_zero_patches,
                            [&]( const size_t patch, const Eigen::VectorXd& p ) {
                                return layoutGlobalPoint( layout, patch, p );
                            } );
        const HierarchicalMultiPatchSplineSpace h1 = buildHierarchicalSplineSpace( hierarchy.levels, hierarchy.leaves );
        const VectorConformingHierarchicalMultiPatchSplineSpace hdiv = buildHDivHierarchicalMultiPatchSplineSpace( h1 );
        const VectorConformingHierarchicalMultiPatchSplineSpace hcurl =
            buildHCurlHierarchicalMultiPatchSplineSpace( h1 );
        std::vector<Eigen::VectorXd> points;
        for( const double xi : { 0.17, 0.42, 0.83 } ) points.push_back( Eigen::VectorXd::Constant( 1, xi ) );
        using Space = std::tuple<std::string, const SplineSpace*, TraceKind>;
        std::vector<std::pair<std::string, ConformityOutcome>> out;
        for( const auto& [name, ss, kind] : { Space{ "H1", &h1, TraceKind::H1 },
                                              Space{ "H(div)", &hdiv, TraceKind::HDiv },
                                              Space{ "H(curl)", &hcurl, TraceKind::HCurl } } )
            out.emplace_back( name,
                              fragmentTraceConformity( layout.name + ", " + name + ": ", *ss, kind, geom, points ) );
        return out;
    }

    Cell patchTopCellAt3d( const MultiPatchCombinatorialMap& cmap,
                           const size_t patch_id,
                           const std::array<size_t, 3>& element )
    {
        const FullyUnflattenedDart unflat( { Dart( element[0] ), Dart( element[1] ), Dart( element[2] ) } );
        return Cell( cmap.toGlobalDart( patch_id, flattenFull( *cmap.constituents().at( patch_id ), unflat ) ),
                     cmap.dim() );
    }

    // Two 2x2x2 unit cubes, patch 0's s-max side glued to patch 1's s-min
    // side.  Patch 0 keeps (1,0,0) coarse and refines the rest; patch 1
    // refines only the element facing it, so each side is finer somewhere.
    struct SplitFaceLayout
    {
        std::string name;
        TPPermutation permutation;
        std::array<size_t, 3> patch_one_refined;
        std::function<Eigen::VectorXd( const Eigen::VectorXd& )> patch_one_global;
    };

    std::vector<SplitFaceLayout> splitFaceLayouts()
    {
        const auto aligned = []( const Eigen::VectorXd& p ) {
            return Eigen::VectorXd( Eigen::Vector3d( 1.0 + p( 0 ), p( 1 ), p( 2 ) ) );
        };
        // Patch 1's (t, u) meets patch 0's (1 - u, t): a quarter turn about s.
        const auto quarter_turn = []( const Eigen::VectorXd& p ) {
            return Eigen::VectorXd( Eigen::Vector3d( 1.0 + p( 0 ), p( 2 ), 1.0 - p( 1 ) ) );
        };
        return { { "3D aligned", TPPermutation::ZeroToZero, { 0, 0, 0 }, aligned },
                 { "3D quarter turn", TPPermutation::ZeroToOne, { 0, 1, 0 }, quarter_turn } };
    }

    LayoutHierarchy splitFaceHierarchy( const SplitFaceLayout& layout )
    {
        const double ptol = 1e-10;
        const ConstituentSide first{ 0, ElementSide( 0, false ).sideId() };
        const ConstituentSide second{ 1, ElementSide( 0, true ).sideId() };
        const InternalConnectionsMap connections = { { first, { layout.permutation, second } },
                                                     { second, { layout.permutation, first } } };
        LayoutHierarchy out;
        for( const KnotVector& kv : { KnotVector( { 0, 0, 0, 0.5, 1, 1, 1 }, ptol ),
                                      KnotVector( { 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1 }, ptol ) } )
        {
            const auto patch = makePatch( { kv, kv, kv }, { 2, 2, 2 } );
            if( out.levels.empty() ) out.level_zero_patches = { patch, patch };
            out.levels.push_back( std::make_shared<const MultiPatchSplineSpace>(
                buildH1MultiPatchSplineSpace( { patch, patch }, connections ) ) );
        }
        out.leaves.resize( out.levels.size() );
        const std::array<size_t, 3> patch_zero_coarse{ 1, 0, 0 };
        for( size_t patch = 0; patch < 2; patch++ )
            for( size_t i = 0; i < 2; i++ )
                for( size_t j = 0; j < 2; j++ )
                    for( size_t k = 0; k < 2; k++ )
                    {
                        const std::array<size_t, 3> e{ i, j, k };
                        if( patch == 0 ? e == patch_zero_coarse : e != layout.patch_one_refined )
                        {
                            out.leaves.at( 0 ).push_back(
                                patchTopCellAt3d( levelCMap( *out.levels.at( 0 ) ), patch, e ) );
                            continue;
                        }
                        for( size_t a = 0; a < 2; a++ )
                            for( size_t b = 0; b < 2; b++ )
                                for( size_t c = 0; c < 2; c++ )
                                    out.leaves.at( 1 ).push_back(
                                        patchTopCellAt3d( levelCMap( *out.levels.at( 1 ) ),
                                                          patch,
                                                          { 2 * i + a, 2 * j + b, 2 * k + c } ) );
                    }
        return out;
    }

    std::vector<std::pair<std::string, ConformityOutcome>> splitFaceTraceConformity( const SplitFaceLayout& layout )
    {
        const LayoutHierarchy hierarchy = splitFaceHierarchy( layout );
        const SplineGeometry geom =
            splineGeometry( hierarchy.levels.front(),
                            hierarchy.level_zero_patches,
                            [&]( const size_t patch, const Eigen::VectorXd& p ) {
                                return patch == 0 ? p : layout.patch_one_global( p );
                            } );
        const HierarchicalMultiPatchSplineSpace h1 = buildHierarchicalSplineSpace( hierarchy.levels, hierarchy.leaves );
        const VectorConformingHierarchicalMultiPatchSplineSpace hdiv = buildHDivHierarchicalMultiPatchSplineSpace( h1 );
        const VectorConformingHierarchicalMultiPatchSplineSpace hcurl =
            buildHCurlHierarchicalMultiPatchSplineSpace( h1 );
        const std::vector<Eigen::VectorXd> points = {
            Eigen::Vector2d( 0.17, 0.62 ), Eigen::Vector2d( 0.71, 0.29 ), Eigen::Vector2d( 0.38, 0.88 ) };
        using Space = std::tuple<std::string, const SplineSpace*, TraceKind>;
        std::vector<std::pair<std::string, ConformityOutcome>> out;
        for( const auto& [name, ss, kind] : { Space{ "H1", &h1, TraceKind::H1 },
                                              Space{ "H(div)", &hdiv, TraceKind::HDiv },
                                              Space{ "H(curl)", &hcurl, TraceKind::HCurl } } )
            out.emplace_back( name,
                              fragmentTraceConformity( layout.name + ", " + name + ": ", *ss, kind, geom, points ) );
        return out;
    }
}

TEST_CASE( "Physical traces agree on patch-interface fragments under asymmetric leaves",
           "[shared_interface_refinement]" )
{
    // The preconditions case's hand-counted fragments per layout.
    const std::vector<size_t> pairs = { 1, 1, 2, 2, 2, 2, 3, 3, 4, 2, 2 };
    const std::vector<TwoPatchLayout> layouts = interfaceLayouts();
    REQUIRE( layouts.size() == pairs.size() );
    std::vector<std::string> problems;
    for( size_t i = 0; i < layouts.size(); i++ )
        for( const auto& [name, o] : layoutTraceConformity( layouts.at( i ) ) )
        {
            INFO( layouts.at( i ).name << ", " << name );
            CHECK( o.records == pairs.at( i ) );
            CHECK( o.nonzero_shared_trace_observations > 0 );
            if( name != "H1" ) CHECK( o.full_vector_jump_observations > 0 );
            problems.insert( problems.end(), o.problems.begin(), o.problems.end() );
        }
    INFO( joined( problems ) );
    CHECK( problems.empty() );
}

TEST_CASE( "Physical traces agree on 3D patch-interface fragments under asymmetric leaves",
           "[shared_interface_refinement]" )
{
    std::vector<std::string> problems;
    for( const SplitFaceLayout& layout : splitFaceLayouts() )
        for( const auto& [name, o] : splitFaceTraceConformity( layout ) )
        {
            INFO( layout.name << ", " << name );
            // Four interface cells, each split into four fragments.
            CHECK( o.records == 16 );
            CHECK( o.nonzero_shared_trace_observations > 0 );
            if( name != "H1" ) CHECK( o.full_vector_jump_observations > 0 );
            problems.insert( problems.end(), o.problems.begin(), o.problems.end() );
        }
    INFO( joined( problems ) );
    CHECK( problems.empty() );
}

// Item 1 step 8: the orientation and refinement-direction matrix.  Each patch
// is a unit cube placed by a rotation (a signed permutation of determinant +1)
// and a shift.  Expected pairs and fragments come from the placed leaf boxes.
// A 3D gluing's permutation is the one the rule restated from
// Stage1_InterfaceTransform_test gives; no production transform is read.
namespace
{
    using Rotation = std::vector<std::vector<int>>; // [placed axis][patch axis]
    using LeafKey = std::vector<size_t>;            // level, patch, element per axis

    std::vector<Rotation> rotations( const size_t dim )
    {
        std::vector<size_t> perm( dim );
        std::iota( perm.begin(), perm.end(), 0 );
        std::vector<Rotation> out;
        do
        {
            int perm_sign = 1;
            for( size_t i = 0; i < dim; i++ )
                for( size_t j = i + 1; j < dim; j++ )
                    if( perm.at( i ) > perm.at( j ) ) perm_sign = -perm_sign;
            for( size_t signs = 0; signs < ( size_t( 1 ) << dim ); signs++ )
            {
                Rotation r( dim, std::vector<int>( dim, 0 ) );
                int det = perm_sign;
                for( size_t k = 0; k < dim; k++ )
                {
                    const int sign = ( signs >> k ) & 1 ? -1 : 1;
                    r.at( perm.at( k ) ).at( k ) = sign;
                    det *= sign;
                }
                if( det == 1 ) out.push_back( r );
            }
        } while( std::next_permutation( perm.begin(), perm.end() ) );
        return out;
    }

    struct Placement
    {
        Rotation rotation;
        Eigen::VectorXd shift; // the patch occupies shift + [0, 1]^dim
    };

    Eigen::VectorXd placed( const Placement& p, const Eigen::VectorXd& x )
    {
        Eigen::VectorXd out = p.shift + Eigen::VectorXd::Constant( x.size(), 0.5 );
        for( Eigen::Index g = 0; g < x.size(); g++ )
            for( Eigen::Index k = 0; k < x.size(); k++ ) out( g ) += p.rotation.at( g ).at( k ) * ( x( k ) - 0.5 );
        return out;
    }

    Eigen::VectorXd unplaced( const Placement& p, const Eigen::VectorXd& y )
    {
        Eigen::VectorXd out = Eigen::VectorXd::Constant( y.size(), 0.5 );
        for( Eigen::Index k = 0; k < y.size(); k++ )
            for( Eigen::Index g = 0; g < y.size(); g++ )
                out( k ) += p.rotation.at( g ).at( k ) * ( y( g ) - p.shift( g ) - 0.5 );
        return out;
    }

    // The placed axis along which a patch axis runs, and its sign.
    std::pair<size_t, int> placedAxis( const Rotation& r, const size_t k )
    {
        for( size_t g = 0; g < r.size(); g++ )
            if( r.at( g ).at( k ) != 0 ) return { g, r.at( g ).at( k ) };
        throw std::runtime_error( "not a rotation" );
    }

    // A side's outward normal, placed.
    std::pair<size_t, int> placedNormal( const Rotation& r, const size_t side_id )
    {
        const auto [g, sign] = placedAxis( r, side_id / 2 );
        return { g, side_id % 2 == 1 ? -sign : sign };
    }

    // The tangential axes of a side, ascending.
    std::vector<size_t> sideAxes( const size_t dim, const size_t side_id )
    {
        std::vector<size_t> out;
        for( size_t k = 0; k < dim; k++ )
            if( k != side_id / 2 ) out.push_back( k );
        return out;
    }

    // Side coordinates v, in sideAxes order, placed.
    Eigen::VectorXd placedSidePoint( const Placement& p, const size_t side_id, const Eigen::VectorXd& v )
    {
        const size_t dim = p.rotation.size();
        Eigen::VectorXd x( dim );
        x( side_id / 2 ) = side_id % 2 == 1 ? 0.0 : 1.0;
        const std::vector<size_t> axes = sideAxes( dim, side_id );
        for( size_t d = 0; d < axes.size(); d++ ) x( axes.at( d ) ) = v( d );
        return placed( p, x );
    }

    // Restated from Stage1_InterfaceTransform_test: the side-b point meeting
    // side-a point v is F_b o P o F_a, a flip being 1 - v.  In 2D, P is a
    // reversal and the flip parity decides it.
    Eigen::VectorXd ruleImage( const Eigen::VectorXd& v, const size_t sa, const size_t sb, const TPPermutation p )
    {
        if( v.size() == 1 )
        {
            const auto flips = []( const size_t s ) { return s == 1 or s == 2; };
            const bool reversed = flips( sa ) == flips( sb );
            return Eigen::VectorXd::Constant( 1, reversed ? 1.0 - v( 0 ) : v( 0 ) );
        }
        const auto flips = []( const size_t s ) { return s == 0 or s == 3 or s == 4; };
        Eigen::Vector2d w = v;
        if( flips( sa ) ) w( 0 ) = 1.0 - w( 0 );
        switch( p )
        {
            case TPPermutation::ZeroToZero: w = Eigen::Vector2d( 1.0 - w( 0 ), w( 1 ) ); break;
            case TPPermutation::ZeroToOne: w = Eigen::Vector2d( 1.0 - w( 1 ), 1.0 - w( 0 ) ); break;
            case TPPermutation::ZeroToTwo: w = Eigen::Vector2d( w( 0 ), 1.0 - w( 1 ) ); break;
            case TPPermutation::ZeroToThree: w = Eigen::Vector2d( w( 1 ), w( 0 ) ); break;
            case TPPermutation::Flip1d: throw std::runtime_error( "Flip1d is not a 3D permutation" );
        }
        if( flips( sb ) ) w( 0 ) = 1.0 - w( 0 );
        return w;
    }

    bool placementFollowsRule( const Placement& pa,
                               const size_t sa,
                               const Placement& pb,
                               const size_t sb,
                               const TPPermutation p )
    {
        const size_t rank = pa.rotation.size() - 1;
        for( const Eigen::Vector2d& v : { Eigen::Vector2d( 0.2, 0.7 ), Eigen::Vector2d( 0.9, 0.4 ) } )
        {
            const Eigen::VectorXd u = v.head( rank );
            if( ( placedSidePoint( pa, sa, u ) - placedSidePoint( pb, sb, ruleImage( u, sa, sb, p ) ) ).norm() >
                geom_tol )
                return false;
        }
        return true;
    }

    struct PlacedLayout
    {
        std::string name;
        std::string orientation; // the interface's class
        std::vector<Placement> patches;
        InternalConnectionsMap connections;
        std::set<size_t> refined_axes;                               // placed axes bisected at level 1
        std::vector<std::pair<size_t, std::vector<size_t>>> refined; // patch, placed level-0 element
        size_t expected_records = 0;
    };

    void glue( PlacedLayout& layout, const size_t pa, const size_t sa, const size_t pb, const size_t sb )
    {
        std::vector<TPPermutation> candidates = { TPPermutation::Flip1d };
        if( layout.patches.front().rotation.size() == 3 )
            candidates = { TPPermutation::ZeroToZero,
                           TPPermutation::ZeroToOne,
                           TPPermutation::ZeroToTwo,
                           TPPermutation::ZeroToThree };
        std::vector<TPPermutation> matches;
        for( const TPPermutation p : candidates )
            if( placementFollowsRule( layout.patches.at( pa ), sa, layout.patches.at( pb ), sb, p ) )
                matches.push_back( p );
        if( matches.size() != 1 )
            throw std::runtime_error( layout.name + ": " + std::to_string( matches.size() ) +
                                      " permutations follow the rule" );
        const ConstituentSide a{ pa, sa };
        const ConstituentSide b{ pb, sb };
        layout.connections.emplace( a, std::pair{ matches.front(), b } );
        layout.connections.emplace( b, std::pair{ matches.front(), a } );
    }

    std::string permutationName( const TPPermutation p )
    {
        switch( p )
        {
            case TPPermutation::ZeroToZero: return "ZeroToZero";
            case TPPermutation::ZeroToOne: return "ZeroToOne";
            case TPPermutation::ZeroToTwo: return "ZeroToTwo";
            case TPPermutation::ZeroToThree: return "ZeroToThree";
            case TPPermutation::Flip1d: return "Flip1d";
        }
        return "?";
    }

    // Whether side b's first tangential axis runs along a different placed
    // axis than side a's, and how many tangential axes run opposite ways.
    std::pair<bool, size_t>
        sideClass( const Placement& pa, const size_t sa, const Placement& pb, const size_t sb )
    {
        const size_t dim = pa.rotation.size();
        const std::vector<size_t> ta = sideAxes( dim, sa );
        const std::vector<size_t> tb = sideAxes( dim, sb );
        const bool swapped = placedAxis( pa.rotation, ta.front() ).first != placedAxis( pb.rotation, tb.front() ).first;
        size_t reversed = 0;
        for( const size_t ka : ta )
            for( const size_t kb : tb )
            {
                const auto [ga, sign_a] = placedAxis( pa.rotation, ka );
                const auto [gb, sign_b] = placedAxis( pb.rotation, kb );
                if( ga == gb and sign_a != sign_b ) reversed++;
            }
        return { swapped, reversed };
    }

    // Patch 0 turned so that its side a faces +x; patch 1 turned by each
    // rotation and shifted by one along x, so that one of its sides meets
    // side a.  Level 1 bisects y (and x in 3D), so a 3D face is refined along
    // one tangential axis only.  Patch 0 refines the interface elements
    // except the one at the origin; patch 1 refines only the one facing it.
    std::vector<PlacedLayout> twoPatchPlacedLayouts( const size_t dim )
    {
        const std::vector<Rotation> all = rotations( dim );
        std::vector<PlacedLayout> out;
        for( size_t sa = 0; sa < 2 * dim; sa++ )
        {
            const auto r0 = std::find_if( all.begin(), all.end(), [&]( const Rotation& r ) {
                return placedNormal( r, sa ) == std::pair<size_t, int>{ 0, 1 };
            } );
            for( const Rotation& r1 : all )
            {
                size_t sb = 0;
                while( placedNormal( r1, sb ) != std::pair<size_t, int>{ 0, -1 } ) sb++;
                PlacedLayout layout;
                layout.patches = { { *r0, Eigen::VectorXd::Zero( dim ) }, { r1, Eigen::VectorXd::Unit( dim, 0 ) } };
                glue( layout, 0, sa, 1, sb );
                const auto [swapped, reversed] = sideClass( layout.patches.at( 0 ), sa, layout.patches.at( 1 ), sb );
                std::ostringstream orientation;
                if( dim == 3 )
                    orientation << permutationName( layout.connections.begin()->second.first )
                                << ( swapped ? ", swapped, " : ", axis-preserving, " );
                orientation << reversed << " reversed";
                layout.orientation = orientation.str();
                layout.name = std::to_string( dim ) + "D sides " + std::to_string( sa ) + "/" + std::to_string( sb ) +
                              ", " + layout.orientation;
                layout.refined_axes = dim == 2 ? std::set<size_t>{ 1 } : std::set<size_t>{ 0, 1 };
                for( size_t e = 0; e < ( size_t( 1 ) << ( dim - 1 ) ); e++ )
                {
                    std::vector<size_t> element{ 1 };
                    for( size_t k = 0; k + 1 < dim; k++ ) element.push_back( ( e >> k ) & 1 );
                    if( e != 0 ) layout.refined.push_back( { 0, element } );
                }
                layout.refined.push_back( { 1, std::vector<size_t>( dim, 0 ) } );
                layout.expected_records = size_t( 2 ) << ( dim - 1 );
                out.push_back( layout );
            }
        }
        return out;
    }

    struct PlacedLeaf
    {
        LeafKey key;
        Eigen::VectorXd lo, hi;             // placed box
        Eigen::VectorXd patch_lo, patch_hi; // in patch coordinates
    };

    size_t elementsAlong( const PlacedLayout& layout, const size_t level, const size_t patch, const size_t k )
    {
        const bool bisected =
            layout.refined_axes.contains( placedAxis( layout.patches.at( patch ).rotation, k ).first );
        return level > 0 and bisected ? 4 : 2;
    }

    std::vector<PlacedLeaf> placedLeaves( const PlacedLayout& layout )
    {
        const size_t dim = layout.patches.front().rotation.size();
        std::vector<PlacedLeaf> out;
        for( size_t patch = 0; patch < layout.patches.size(); patch++ )
        {
            const Placement& pl = layout.patches.at( patch );
            for( size_t e = 0; e < ( size_t( 1 ) << dim ); e++ )
            {
                std::vector<size_t> element;
                for( size_t g = 0; g < dim; g++ ) element.push_back( ( e >> g ) & 1 );
                const bool refined = std::find( layout.refined.begin(),
                                                layout.refined.end(),
                                                std::pair{ patch, element } ) != layout.refined.end();
                const size_t n_children = refined ? size_t( 1 ) << layout.refined_axes.size() : 1;
                for( size_t c = 0; c < n_children; c++ )
                {
                    PlacedLeaf leaf;
                    leaf.lo = pl.shift;
                    leaf.hi = pl.shift;
                    size_t bit = 0;
                    for( size_t g = 0; g < dim; g++ )
                    {
                        const bool split = refined and layout.refined_axes.contains( g );
                        const double h = split ? 0.25 : 0.5;
                        leaf.lo( g ) += 0.5 * element.at( g ) + ( split ? h * ( ( c >> bit++ ) & 1 ) : 0.0 );
                        leaf.hi( g ) = leaf.lo( g ) + h;
                    }
                    const Eigen::VectorXd a = unplaced( pl, leaf.lo );
                    const Eigen::VectorXd b = unplaced( pl, leaf.hi );
                    leaf.patch_lo = a.cwiseMin( b );
                    leaf.patch_hi = a.cwiseMax( b );
                    const size_t level = refined ? 1 : 0;
                    leaf.key = { level, patch };
                    for( size_t k = 0; k < dim; k++ )
                        leaf.key.push_back( static_cast<size_t>(
                            std::lround( leaf.patch_lo( k ) * elementsAlong( layout, level, patch, k ) ) ) );
                    out.push_back( leaf );
                }
            }
        }
        return out;
    }

    // The face two placed boxes share, degenerate along its normal; empty
    // unless its measure is positive.
    std::optional<std::pair<Eigen::VectorXd, Eigen::VectorXd>> sharedFace( const PlacedLeaf& a, const PlacedLeaf& b )
    {
        const Eigen::VectorXd lo = a.lo.cwiseMax( b.lo );
        const Eigen::VectorXd hi = a.hi.cwiseMin( b.hi );
        size_t touching = 0;
        for( Eigen::Index g = 0; g < lo.size(); g++ )
        {
            if( std::abs( hi( g ) - lo( g ) ) < geom_tol )
                touching++;
            else if( hi( g ) < lo( g ) )
                return std::nullopt;
        }
        if( touching != 1 ) return std::nullopt;
        return std::pair{ lo, hi };
    }

    Cell patchTopCellNd( const MultiPatchCombinatorialMap& cmap, const size_t patch_id, const LeafKey& key )
    {
        SmallVector<Dart, 3> darts;
        for( size_t k = 2; k < key.size(); k++ ) darts.push_back( Dart( key.at( k ) ) );
        return Cell( cmap.toGlobalDart(
                         patch_id, flattenFull( *cmap.constituents().at( patch_id ), FullyUnflattenedDart( darts ) ) ),
                     cmap.dim() );
    }

    LeafKey levelKey( const MultiPatchCombinatorialMap& level_cmap, const size_t level, const Cell& c )
    {
        const auto [patch, d] = level_cmap.toLocalDart( c.dart() );
        LeafKey out{ level, patch };
        for( const Dart& u : unflattenFull( *level_cmap.constituents().at( patch ), d ).unflat_darts )
            out.push_back( u.id() );
        return out;
    }

    // Throws unless the element is named by a leaf dart.
    LeafKey leafKeyOf( const HierarchicalMultiPatchCombinatorialMap& cmap, const Cell& element )
    {
        const auto [level, level_d] = cmap.unrefinedAncestorDartOfCell( element );
        return levelKey( *cmap.refinementLevels().at( level ), level, Cell( level_d, cmap.dim() ) );
    }

    std::string describe( const LeafKey& k )
    {
        std::ostringstream out;
        out << "p" << k.at( 1 ) << " L" << k.at( 0 ) << " (";
        for( size_t i = 2; i < k.size(); i++ ) out << ( i > 2 ? "," : "" ) << k.at( i );
        out << ")";
        return out.str();
    }

    LayoutHierarchy placedHierarchy( const PlacedLayout& layout, const std::vector<PlacedLeaf>& leaves )
    {
        const double ptol = 1e-10;
        const size_t dim = layout.patches.front().rotation.size();
        LayoutHierarchy out;
        out.leaves.resize( 2 );
        for( size_t level = 0; level < 2; level++ )
        {
            std::vector<std::shared_ptr<const TPSplineSpace>> patches;
            for( size_t patch = 0; patch < layout.patches.size(); patch++ )
            {
                SmallVector<KnotVector, 3> kvs;
                for( size_t k = 0; k < dim; k++ )
                {
                    const size_t n = elementsAlong( layout, level, patch, k );
                    std::vector<double> knots{ 0, 0 };
                    for( size_t i = 0; i <= n; i++ ) knots.push_back( double( i ) / double( n ) );
                    knots.insert( knots.end(), { 1, 1 } );
                    kvs.push_back( KnotVector( knots, ptol ) );
                }
                patches.push_back( makePatch( kvs, SmallVector<size_t, 3>( dim, 2 ) ) );
            }
            if( level == 0 ) out.level_zero_patches = patches;
            out.levels.push_back( std::make_shared<const MultiPatchSplineSpace>(
                buildH1MultiPatchSplineSpace( patches, layout.connections ) ) );
        }
        for( const PlacedLeaf& leaf : leaves )
            out.leaves.at( leaf.key.at( 0 ) )
                .push_back(
                    patchTopCellNd( levelCMap( *out.levels.at( leaf.key.at( 0 ) ) ), leaf.key.at( 1 ), leaf.key ) );
        return out;
    }

    struct PlacedOutcome
    {
        std::vector<std::string> problems;
        size_t expected_records = 0;
        std::vector<std::pair<std::string, ConformityOutcome>> traces;
    };

    // Requested leaves are the effective leaves; each record pairs two leaves
    // whose placed boxes share a face, once, and both sides place the
    // fragment's corners on that face's corners.  Then physical traces.
    PlacedOutcome placedLayoutOutcome( const PlacedLayout& layout )
    {
        PlacedOutcome out;
        const size_t dim = layout.patches.front().rotation.size();
        const std::string prefix = layout.name + ", ";
        const std::vector<PlacedLeaf> leaves = placedLeaves( layout );
        std::map<LeafKey, const PlacedLeaf*> by_key;
        for( const PlacedLeaf& leaf : leaves )
        {
            if( not by_key.emplace( leaf.key, &leaf ).second )
                out.problems.push_back( prefix + "duplicate leaf " + describe( leaf.key ) );
            for( size_t k = 0; k < dim; k++ )
            {
                const double n = elementsAlong( layout, leaf.key.at( 0 ), leaf.key.at( 1 ), k );
                if( std::abs( leaf.patch_lo( k ) - leaf.key.at( k + 2 ) / n ) > geom_tol or
                    std::abs( leaf.patch_hi( k ) - ( leaf.key.at( k + 2 ) + 1 ) / n ) > geom_tol )
                    out.problems.push_back( prefix + describe( leaf.key ) + " is not an element" );
            }
        }
        std::map<std::pair<LeafKey, LeafKey>, std::pair<Eigen::VectorXd, Eigen::VectorXd>> expected;
        for( const PlacedLeaf& a : leaves )
            for( const PlacedLeaf& b : leaves )
                if( a.key.at( 1 ) < b.key.at( 1 ) )
                    if( const auto face = sharedFace( a, b ) ) expected.emplace( std::pair{ a.key, b.key }, *face );
        out.expected_records = expected.size();

        const LayoutHierarchy hierarchy = placedHierarchy( layout, leaves );
        const HierarchicalMultiPatchSplineSpace h1 = buildHierarchicalSplineSpace( hierarchy.levels, hierarchy.leaves );
        const VectorConformingHierarchicalMultiPatchSplineSpace hdiv = buildHDivHierarchicalMultiPatchSplineSpace( h1 );
        const VectorConformingHierarchicalMultiPatchSplineSpace hcurl =
            buildHCurlHierarchicalMultiPatchSplineSpace( h1 );
        using Space = std::tuple<std::string, const SplineSpace*, TraceKind>;
        const std::vector<Space> spaces = { Space{ "H1", &h1, TraceKind::H1 },
                                            Space{ "H(div)", &hdiv, TraceKind::HDiv },
                                            Space{ "H(curl)", &hcurl, TraceKind::HCurl } };

        std::vector<Eigen::VectorXd> corners;
        for( size_t c = 0; c < ( size_t( 1 ) << ( dim - 1 ) ); c++ )
        {
            Eigen::VectorXd xi( dim - 1 );
            for( size_t d = 0; d + 1 < dim; d++ ) xi( d ) = ( c >> d ) & 1;
            corners.push_back( xi );
        }
        std::vector<Eigen::VectorXd> fragment_points = corners;
        fragment_points.push_back( Eigen::Vector2d( 0.3, 0.7 ).head( dim - 1 ) );

        for( const auto& [name, ss, kind] : spaces )
        {
            const std::string where = prefix + name + ": ";
            const auto& cmap = dynamic_cast<const HierarchicalMultiPatchCombinatorialMap&>(
                ss->basisComplex().parametricAtlas().cmap() );

            // Action (b): leafElements returns the requested leaves.
            std::multiset<LeafKey> effective;
            const std::vector<std::vector<Cell>> leaf_cells = leafElements( cmap );
            for( size_t level = 0; level < leaf_cells.size(); level++ )
                for( const Cell& c : leaf_cells.at( level ) )
                    effective.insert( levelKey( *cmap.refinementLevels().at( level ), level, c ) );
            std::multiset<LeafKey> requested;
            for( const auto& [k, leaf] : by_key ) requested.insert( k );
            if( effective != requested )
                out.problems.push_back( where + "leafElements differs from the requested leaves" );

            std::multiset<std::pair<LeafKey, LeafKey>> actual;
            for( const TraceMeshInterface& iface : patchTraceMeshInterfaces( *ss ) )
            {
                if( iface.type != TraceMeshEntityType::PatchInterface or not iface.second.has_value() )
                {
                    out.problems.push_back( where + "a record is not a two-sided patch interface" );
                    continue;
                }
                std::array<LeafKey, 2> keys;
                const std::array<const TraceMeshSide*, 2> sides = { &iface.first, &*iface.second };
                try
                {
                    for( size_t i = 0; i < 2; i++ ) keys.at( i ) = leafKeyOf( cmap, sides.at( i )->element );
                }
                catch( const std::exception& e )
                {
                    out.problems.push_back( where + "a side is not named by a leaf dart: " + e.what() );
                    continue;
                }
                const std::string text = describe( keys.at( 0 ) ) + " / " + describe( keys.at( 1 ) );
                bool known = true;
                for( size_t i = 0; i < 2; i++ )
                {
                    if( keys.at( i ).at( 0 ) != sides.at( i )->level or
                        keys.at( i ).at( 1 ) != sides.at( i )->patch_id )
                        out.problems.push_back( where + "record fields disagree with " + text );
                    if( not by_key.contains( keys.at( i ) ) )
                    {
                        out.problems.push_back( where + "non-leaf " + describe( keys.at( i ) ) );
                        known = false;
                    }
                }
                const auto pair = keys.at( 0 ).at( 1 ) < keys.at( 1 ).at( 1 ) ? std::pair{ keys.at( 0 ), keys.at( 1 ) }
                                                                              : std::pair{ keys.at( 1 ), keys.at( 0 ) };
                actual.insert( pair );
                const auto face = expected.find( pair );
                if( not known or face == expected.end() ) continue;

                std::array<std::vector<Eigen::VectorXd>, 2> images;
                try
                {
                    for( size_t i = 0; i < 2; i++ )
                    {
                        const TraceMeshSide& side = *sides.at( i );
                        const PlacedLeaf& leaf = *by_key.at( keys.at( i ) );
                        for( const Eigen::VectorXd& xi : fragment_points )
                        {
                            const Eigen::VectorXd y = embedTraceFragmentPoint( side.embedding, xi );
                            Eigen::VectorXd x( dim );
                            x( side.side.axis ) = side.side.lower ? leaf.patch_lo( side.side.axis )
                                                                  : leaf.patch_hi( side.side.axis );
                            const std::vector<size_t> axes = sideAxes( dim, side.side.sideId() );
                            for( size_t d = 0; d < axes.size(); d++ )
                                x( axes.at( d ) ) =
                                    leaf.patch_lo( axes.at( d ) ) +
                                    y( d ) * ( leaf.patch_hi( axes.at( d ) ) - leaf.patch_lo( axes.at( d ) ) );
                            images.at( i ).push_back( placed( layout.patches.at( side.patch_id ), x ) );
                        }
                    }
                }
                catch( const std::exception& e )
                {
                    out.problems.push_back( where + text + ": " + e.what() );
                    continue;
                }
                for( size_t c = 0; c < fragment_points.size(); c++ )
                    if( ( images.at( 0 ).at( c ) - images.at( 1 ).at( c ) ).norm() > geom_tol )
                        out.problems.push_back( where + text + ": the sides place a fragment point apart" );
                const auto& [lo, hi] = face->second;
                std::set<std::vector<long>> distinct;
                for( size_t c = 0; c < corners.size(); c++ )
                {
                    const Eigen::VectorXd& x = images.at( 0 ).at( c );
                    std::vector<long> corner;
                    for( size_t g = 0; g < dim; g++ )
                    {
                        const bool at_lo = std::abs( x( g ) - lo( g ) ) < geom_tol;
                        if( not at_lo and std::abs( x( g ) - hi( g ) ) >= geom_tol )
                            out.problems.push_back( where + text + ": a fragment corner is off the shared face" );
                        corner.push_back( at_lo ? 0 : 1 );
                    }
                    distinct.insert( corner );
                }
                if( distinct.size() != corners.size() )
                    out.problems.push_back( where + text + ": the fragment is not the shared face" );
            }
            for( const auto& pair : actual )
            {
                const std::string text = describe( pair.first ) + " / " + describe( pair.second );
                if( actual.count( pair ) > 1 ) out.problems.push_back( where + "duplicate " + text );
                if( not expected.contains( pair ) ) out.problems.push_back( where + "unexpected " + text );
            }
            for( const auto& [pair, face] : expected )
                if( not actual.contains( pair ) )
                    out.problems.push_back( where + "missing " + describe( pair.first ) + " / " +
                                            describe( pair.second ) );
        }

        const SplineGeometry geom =
            splineGeometry( hierarchy.levels.front(),
                            hierarchy.level_zero_patches,
                            [&]( const size_t patch, const Eigen::VectorXd& p ) {
                                return placed( layout.patches.at( patch ), p );
                            } );
        std::vector<Eigen::VectorXd> points;
        if( dim == 2 )
            for( const double xi : { 0.17, 0.42, 0.83 } ) points.push_back( Eigen::VectorXd::Constant( 1, xi ) );
        else
            points = { Eigen::Vector2d( 0.17, 0.62 ), Eigen::Vector2d( 0.71, 0.29 ), Eigen::Vector2d( 0.38, 0.88 ) };
        for( const auto& [name, ss, kind] : spaces )
            out.traces.emplace_back( name, fragmentTraceConformity( prefix + name + ": ", *ss, kind, geom, points ) );
        return out;
    }
}

TEST_CASE( "Placed two-patch layouts: every side pair and permutation once", "[shared_interface_refinement]" )
{
    std::map<std::string, size_t> classes;
    for( const size_t dim : { size_t( 2 ), size_t( 3 ) } )
    {
        std::set<std::tuple<size_t, size_t, TPPermutation>> triples;
        const std::vector<PlacedLayout> layouts = twoPatchPlacedLayouts( dim );
        for( const PlacedLayout& layout : layouts )
        {
            const auto& [a, glued] = *layout.connections.begin();
            triples.insert( { a.side_id, glued.second.side_id, glued.first } );
            classes[std::to_string( dim ) + "D " + layout.orientation]++;
        }
        CHECK( layouts.size() == ( dim == 2 ? 16 : 144 ) );
        CHECK( triples.size() == layouts.size() );
    }
    const std::map<std::string, size_t> expected = { { "2D 0 reversed", 8 },
                                                     { "2D 1 reversed", 8 },
                                                     { "3D ZeroToOne, swapped, 0 reversed", 9 },
                                                     { "3D ZeroToOne, swapped, 1 reversed", 18 },
                                                     { "3D ZeroToOne, swapped, 2 reversed", 9 },
                                                     { "3D ZeroToThree, swapped, 0 reversed", 9 },
                                                     { "3D ZeroToThree, swapped, 1 reversed", 18 },
                                                     { "3D ZeroToThree, swapped, 2 reversed", 9 },
                                                     { "3D ZeroToTwo, axis-preserving, 1 reversed", 18 },
                                                     { "3D ZeroToTwo, axis-preserving, 2 reversed", 18 },
                                                     { "3D ZeroToZero, axis-preserving, 0 reversed", 18 },
                                                     { "3D ZeroToZero, axis-preserving, 1 reversed", 18 } };
    CHECK( classes == expected );
}

TEST_CASE( "Placed two-patch layouts: leaves, fragments and physical traces under asymmetric leaves",
           "[shared_interface_refinement]" )
{
    std::vector<std::string> problems;
    for( const size_t dim : { size_t( 2 ), size_t( 3 ) } )
        for( const PlacedLayout& layout : twoPatchPlacedLayouts( dim ) )
        {
            INFO( layout.name );
            const PlacedOutcome o = placedLayoutOutcome( layout );
            CHECK( o.expected_records == layout.expected_records );
            problems.insert( problems.end(), o.problems.begin(), o.problems.end() );
            for( const auto& [name, t] : o.traces )
            {
                INFO( name );
                CHECK( t.records == layout.expected_records );
                CHECK( t.nonzero_shared_trace_observations > 0 );
                if( name != "H1" ) CHECK( t.full_vector_jump_observations > 0 );
                problems.insert( problems.end(), t.problems.begin(), t.problems.end() );
            }
        }
    INFO( joined( problems ) );
    CHECK( problems.empty() );
}

// Step 8, second half: four unit cubes around a vertex (2D) or an edge (3D),
// patch p shifted by cycleShift and glued on every touching face.  Patches 0
// and 2 refine their elements at the junction; 1 and 3 stay coarse.  The
// rotations and interface signatures were selected offline and are pinned.
namespace
{
    const std::array<std::pair<size_t, size_t>, 4> cycle_pairs = { { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 0, 3 } } };
    const std::set<size_t> cycle_refined_patches = { 0, 2 };

    Eigen::VectorXd cycleShift( const size_t dim, const size_t patch )
    {
        Eigen::VectorXd out = Eigen::VectorXd::Zero( dim );
        out( 0 ) = patch == 1 or patch == 2 ? 1.0 : 0.0;
        out( 1 ) = patch >= 2 ? 1.0 : 0.0;
        return out;
    }

    struct CycleInterface
    {
        size_t pa, sa, pb, sb;
        TPPermutation permutation;
        bool swapped;
        size_t reversed;
    };

    struct CycleFixture
    {
        std::string name;
        std::vector<Rotation> rotations;        // by patch
        std::vector<CycleInterface> interfaces; // in cycle_pairs order
    };

    std::vector<CycleFixture> cycleFixtures()
    {
        using P = TPPermutation;
        const Rotation i2 = { { 1, 0 }, { 0, 1 } };
        const Rotation half2 = { { -1, 0 }, { 0, -1 } };
        const Rotation i3 = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        const Rotation half_y = { { -1, 0, 0 }, { 0, 1, 0 }, { 0, 0, -1 } };
        const Rotation third = { { 0, -1, 0 }, { 0, 0, -1 }, { 1, 0, 0 } };
        return { { "2D aligned",
                   { i2, i2, i2, i2 },
                   { { 0, 0, 1, 1, P::Flip1d, false, 0 },
                     { 1, 2, 2, 3, P::Flip1d, false, 0 },
                     { 2, 1, 3, 0, P::Flip1d, false, 0 },
                     { 0, 2, 3, 3, P::Flip1d, false, 0 } } },
                 { "2D patch 2 half turn",
                   { i2, i2, half2, i2 },
                   { { 0, 0, 1, 1, P::Flip1d, false, 0 },
                     { 1, 2, 2, 2, P::Flip1d, false, 1 },
                     { 2, 0, 3, 0, P::Flip1d, false, 1 },
                     { 0, 2, 3, 3, P::Flip1d, false, 0 } } },
                 { "3D aligned",
                   { i3, i3, i3, i3 },
                   { { 0, 0, 1, 1, P::ZeroToZero, false, 0 },
                     { 1, 2, 2, 3, P::ZeroToZero, false, 0 },
                     { 2, 1, 3, 0, P::ZeroToZero, false, 0 },
                     { 0, 2, 3, 3, P::ZeroToZero, false, 0 } } },
                 { "3D patch 3 half turn about y",
                   { i3, i3, i3, half_y },
                   { { 0, 0, 1, 1, P::ZeroToZero, false, 0 },
                     { 1, 2, 2, 3, P::ZeroToZero, false, 0 },
                     { 2, 1, 3, 1, P::ZeroToTwo, false, 1 },
                     { 0, 2, 3, 3, P::ZeroToTwo, false, 2 } } },
                 { "3D patch 2 third turn",
                   { i3, i3, third, i3 },
                   { { 0, 0, 1, 1, P::ZeroToZero, false, 0 },
                     { 1, 2, 2, 4, P::ZeroToOne, true, 1 },
                     { 2, 2, 3, 0, P::ZeroToThree, true, 1 },
                     { 0, 2, 3, 3, P::ZeroToZero, false, 0 } } } };
    }

    // Level 1 bisects placed x and y, so a 3D interface is refined along one
    // tangential axis.  The refined patches split every element at the
    // junction.
    PlacedLayout cycleLayout( const CycleFixture& fixture )
    {
        const size_t dim = fixture.rotations.front().size();
        PlacedLayout layout;
        layout.name = fixture.name;
        for( size_t p = 0; p < 4; p++ ) layout.patches.push_back( { fixture.rotations.at( p ), cycleShift( dim, p ) } );
        for( const CycleInterface& i : fixture.interfaces ) glue( layout, i.pa, i.sa, i.pb, i.sb );
        layout.refined_axes = { 0, 1 };
        for( const size_t p : cycle_refined_patches )
            for( const size_t z : dim == 2 ? std::vector<size_t>{ 0 } : std::vector<size_t>{ 0, 1 } )
            {
                std::vector<size_t> element( dim, p == 0 ? 1 : 0 );
                if( dim == 3 ) element.at( 2 ) = z;
                layout.refined.push_back( { p, element } );
            }
        layout.expected_records = dim == 2 ? 12 : 24;
        return layout;
    }

    struct CyclePins
    {
        std::vector<std::string> problems;
        std::map<std::string, std::map<size_t, size_t>> spans; // space, patches spanned, classes
    };

    struct ClassEntry
    {
        size_t patch;
        size_t component;
        bool orientation;
    };

    // Patch-local entries of each global function.  Local vector functions are
    // component-major, so the component comes from the scalar bases' sizes.
    std::map<FunctionId, std::vector<ClassEntry>>
        vectorClasses( const VectorConformingHierarchicalMultiPatchSplineSpace& ss )
    {
        std::map<FunctionId, std::vector<ClassEntry>> out;
        for( size_t p = 0; p < ss.functionIdMap().size(); p++ )
        {
            const auto& bases = ss.constituents().at( p )->scalarBases();
            size_t component = 0;
            size_t bound = bases.at( 0 )->numFunctions();
            for( size_t i = 0; i < ss.functionIdMap().at( p ).size(); i++ )
            {
                while( i >= bound ) bound += bases.at( ++component )->numFunctions();
                const auto& [global, orientation] = ss.functionIdMap().at( p ).at( i );
                out[global].push_back( { p, component, orientation } );
            }
        }
        return out;
    }

    // Pins at the junction that do not rest on splineGeometry.
    CyclePins cyclePins( const PlacedLayout& layout )
    {
        CyclePins out;
        const size_t dim = layout.patches.front().rotation.size();
        const std::string prefix = layout.name + ", ";
        const std::set<std::pair<size_t, size_t>> adjacent( cycle_pairs.begin(), cycle_pairs.end() );
        const LayoutHierarchy hierarchy = placedHierarchy( layout, placedLeaves( layout ) );

        // Four declared interfaces, read back from the level-0 map; cycle rank
        // E - V + C is one.
        const InternalConnectionsMap& connections = levelCMap( *hierarchy.levels.front() ).connections();
        std::set<std::pair<size_t, size_t>> glued;
        for( const auto& [a, c] : connections )
            glued.insert( std::minmax( a.constituent_id, c.second.constituent_id ) );
        std::vector<size_t> label = { 0, 1, 2, 3 };
        for( size_t pass = 0; pass < 4; pass++ )
            for( const auto& [a, b] : glued ) label.at( a ) = label.at( b ) = std::min( label.at( a ), label.at( b ) );
        const size_t components = std::set<size_t>( label.begin(), label.end() ).size();
        if( connections.size() != 8 or glued != adjacent )
            out.problems.push_back( prefix + "the declared interfaces are not the four cycle edges" );
        if( glued.size() + components != 4 + 1 ) out.problems.push_back( prefix + "the cycle rank is not one" );

        // The rule's chart from patch a into patch b's coordinates, extended
        // across the shared side; composed around the cycle it is the identity.
        // It is affine, so dim + 1 affinely independent fixed points suffice.
        const auto connection = [&]( const size_t a, const size_t b ) {
            for( const auto& [from, c] : connections )
                if( from.constituent_id == a and c.second.constituent_id == b )
                    return std::tuple{ from.side_id, c.second.side_id, c.first };
            throw std::runtime_error( "patches " + std::to_string( a ) + " and " + std::to_string( b ) +
                                      " are not glued" );
        };
        const auto chart = [&]( const size_t a, const size_t b, const Eigen::VectorXd& x ) {
            const auto [sa, sb, p] = connection( a, b );
            const double depth = sa % 2 == 1 ? x( sa / 2 ) : 1.0 - x( sa / 2 );
            const std::vector<size_t> ta = sideAxes( dim, sa );
            const std::vector<size_t> tb = sideAxes( dim, sb );
            Eigen::VectorXd v( dim - 1 );
            for( size_t d = 0; d + 1 < dim; d++ ) v( d ) = x( ta.at( d ) );
            const Eigen::VectorXd w = ruleImage( v, sa, sb, p );
            Eigen::VectorXd y( dim );
            y( sb / 2 ) = sb % 2 == 1 ? -depth : 1.0 + depth;
            for( size_t d = 0; d + 1 < dim; d++ ) y( tb.at( d ) ) = w( d );
            return y;
        };
        const std::vector<Eigen::Vector3d> starts = {
            { 0.13, 0.71, 0.37 }, { 0.92, 0.18, 0.55 }, { 0.38, 0.94, 0.81 }, { 0.71, 0.62, 0.09 } };
        Eigen::MatrixXd spread( dim, dim );
        for( size_t k = 0; k < dim; k++ ) spread.col( k ) = ( starts.at( k + 1 ) - starts.front() ).head( dim );
        if( std::abs( spread.determinant() ) < 0.1 )
            out.problems.push_back( prefix + "the chart sample points are affinely dependent" );
        try
        {
            for( size_t k = 0; k <= dim; k++ )
            {
                const Eigen::VectorXd start = starts.at( k ).head( dim );
                Eigen::VectorXd x = start;
                for( size_t p = 0; p < 4; p++ ) x = chart( p, ( p + 1 ) % 4, x );
                if( ( x - start ).norm() > geom_tol )
                    out.problems.push_back( prefix + "the composed interface charts are not the identity" );
            }
        }
        catch( const std::exception& e )
        {
            out.problems.push_back( prefix + e.what() );
            return out;
        }

        const HierarchicalMultiPatchSplineSpace h1 = buildHierarchicalSplineSpace( hierarchy.levels, hierarchy.leaves );
        const VectorConformingHierarchicalMultiPatchSplineSpace hdiv = buildHDivHierarchicalMultiPatchSplineSpace( h1 );
        const VectorConformingHierarchicalMultiPatchSplineSpace hcurl =
            buildHCurlHierarchicalMultiPatchSplineSpace( h1 );
        const std::vector<std::pair<std::string, const SplineSpace*>> spaces = {
            { "H1", &h1 }, { "H(div)", &hdiv }, { "H(curl)", &hcurl } };

        // Leaves per patch and level: refined patches split their junction
        // elements, and nothing propagates into the coarse patches.
        std::map<std::pair<size_t, size_t>, size_t> expected_leaves;
        for( size_t p = 0; p < 4; p++ ) expected_leaves[{ p, 0 }] = size_t( 1 ) << dim;
        for( const auto& [p, element] : layout.refined )
        {
            expected_leaves[{ p, 0 }]--;
            expected_leaves[{ p, 1 }] += size_t( 1 ) << layout.refined_axes.size();
        }
        // Each interface: two (3D: four) fine faces against coarse ones, and one
        // (3D: two) coarse pair; the fine side is a refined patch.
        std::map<std::pair<size_t, size_t>, std::pair<size_t, size_t>> expected_records;
        for( const auto& pair : cycle_pairs ) expected_records[pair] = dim == 2 ? std::pair{ 3, 2 } : std::pair{ 6, 4 };

        for( const auto& [name, ss] : spaces )
        {
            const std::string where = prefix + name + ": ";
            const auto& cmap = dynamic_cast<const HierarchicalMultiPatchCombinatorialMap&>(
                ss->basisComplex().parametricAtlas().cmap() );
            std::map<std::pair<size_t, size_t>, size_t> leaves;
            const std::vector<std::vector<Cell>> leaf_cells = leafElements( cmap );
            for( size_t level = 0; level < leaf_cells.size(); level++ )
                for( const Cell& c : leaf_cells.at( level ) )
                    leaves[{ levelKey( *cmap.refinementLevels().at( level ), level, c ).at( 1 ), level }]++;
            if( leaves != expected_leaves ) out.problems.push_back( where + "leaves per patch and level differ" );

            std::map<std::pair<size_t, size_t>, std::pair<size_t, size_t>> records; // all, coarse-fine
            for( const TraceMeshInterface& iface : patchTraceMeshInterfaces( *ss ) )
            {
                if( not iface.second.has_value() ) continue; // placedLayoutOutcome reports it
                const TraceMeshSide& a = iface.first;
                const TraceMeshSide& b = *iface.second;
                auto& count = records[std::minmax( a.patch_id, b.patch_id )];
                count.first++;
                if( a.level == b.level ) continue;
                count.second++;
                if( not cycle_refined_patches.contains( ( a.level > b.level ? a : b ).patch_id ) )
                    out.problems.push_back( where + "a coarse patch is the fine side of " + sideText( a ) + " / " +
                                            sideText( b ) );
            }
            if( records != expected_records ) out.problems.push_back( where + "records per patch pair differ" );
        }

        // Classes: the patches a global function spans, with at most one local
        // function per patch.  Two-patch classes are adjacent.  A vector class
        // keeps one placed axis, and two entries' orientations agree exactly
        // when their components point the same way.
        const auto spanned = [&]( const std::string& where, const std::set<size_t>& patches ) {
            if( patches.size() == 2 and not adjacent.contains( { *patches.begin(), *patches.rbegin() } ) )
                out.problems.push_back( where + "a class spans diagonal patches" );
            if( patches.size() == 3 ) out.problems.push_back( where + "a class spans three patches" );
        };
        std::map<FunctionId, std::set<size_t>> h1_classes;
        for( size_t p = 0; p < h1.functionIdMap().size(); p++ )
            for( const FunctionId& g : h1.functionIdMap().at( p ) )
                if( not h1_classes[g].insert( p ).second )
                    out.problems.push_back( prefix + "H1: a class holds two functions of one patch" );
        for( const auto& [g, patches] : h1_classes )
        {
            out.spans["H1"][patches.size()]++;
            spanned( prefix + "H1: ", patches );
        }

        using VectorSpace = std::pair<std::string, const VectorConformingHierarchicalMultiPatchSplineSpace*>;
        for( const auto& [name, ss] : { VectorSpace{ "H(div)", &hdiv }, VectorSpace{ "H(curl)", &hcurl } } )
        {
            const std::string where = prefix + name + ": ";
            const bool div = name == "H(div)";
            for( const auto& [g, entries] : vectorClasses( *ss ) )
            {
                std::set<size_t> patches;
                for( const ClassEntry& e : entries ) patches.insert( e.patch );
                if( patches.size() != entries.size() )
                    out.problems.push_back( where + "a class holds two functions of one patch" );
                out.spans[name][patches.size()]++;
                spanned( where, patches );
                if( patches.size() < 2 ) continue;

                const auto axis = [&]( const ClassEntry& e ) {
                    return placedAxis( layout.patches.at( e.patch ).rotation, e.component );
                };
                for( const ClassEntry& e : entries )
                {
                    if( axis( e ).first != axis( entries.front() ).first )
                        out.problems.push_back( where + "a class mixes placed axes" );
                    if( ( e.orientation == entries.front().orientation ) !=
                        ( axis( e ).second == axis( entries.front() ).second ) )
                        out.problems.push_back( where + "a relative sign differs from the placement" );
                }
                // The placed axis along which the patches' shifts differ.
                const auto normal = [&]( const size_t a, const size_t b ) {
                    const Eigen::VectorXd d = layout.patches.at( b ).shift - layout.patches.at( a ).shift;
                    Eigen::Index g = 0;
                    d.cwiseAbs().maxCoeff( &g );
                    return size_t( g );
                };
                if( patches.size() == 2 )
                {
                    const size_t n = normal( *patches.begin(), *patches.rbegin() );
                    if( ( axis( entries.front() ).first == n ) != div )
                        out.problems.push_back( where + "a two-patch class has the wrong placed axis" );
                    continue;
                }
                if( div or dim == 2 )
                {
                    out.problems.push_back( where + "a class spans four patches" );
                    continue;
                }
                if( axis( entries.front() ).first != 2 )
                    out.problems.push_back( where + "a four-patch class is not along the edge" );

                // Around the cycle: the rule carries each patch's edge component
                // to the next one's, with sign sigma; the signs have even parity
                // and match the orientations.
                std::map<size_t, ClassEntry> by_patch;
                for( const ClassEntry& e : entries ) by_patch.emplace( e.patch, e );
                int parity = 1;
                for( size_t p = 0; p < 4; p++ )
                {
                    const ClassEntry& ea = by_patch.at( p );
                    const ClassEntry& eb = by_patch.at( ( p + 1 ) % 4 );
                    const auto [sa, sb, perm] = connection( p, ( p + 1 ) % 4 );
                    const std::vector<size_t> ta = sideAxes( dim, sa );
                    const std::vector<size_t> tb = sideAxes( dim, sb );
                    const auto i = std::find( ta.begin(), ta.end(), ea.component ) - ta.begin();
                    if( i == static_cast<std::ptrdiff_t>( ta.size() ) )
                    {
                        out.problems.push_back( where + "an edge component is normal to its side" );
                        continue;
                    }
                    Eigen::VectorXd v = Eigen::Vector2d( 0.3, 0.4 );
                    const Eigen::VectorXd w0 = ruleImage( v, sa, sb, perm );
                    v( i ) += 0.25;
                    const Eigen::VectorXd dw = ruleImage( v, sa, sb, perm ) - w0;
                    Eigen::Index j = 0;
                    dw.cwiseAbs().maxCoeff( &j );
                    const int sigma = dw( j ) > 0 ? 1 : -1;
                    parity *= sigma;
                    if( tb.at( j ) != eb.component )
                        out.problems.push_back( where + "the rule does not carry the edge component across" );
                    if( ( ea.orientation == eb.orientation ) != ( sigma == 1 ) )
                        out.problems.push_back( where + "an orientation differs from the rule's sign" );
                }
                if( parity != 1 ) out.problems.push_back( where + "the edge signs have odd parity" );
            }
        }
        return out;
    }
}

TEST_CASE( "Placed four-patch cycles: rotations and interface signatures", "[shared_interface_refinement]" )
{
    std::set<TPPermutation> permutations;
    std::set<std::pair<std::string, bool>> owned; // swapped or reversed, refined patch first
    for( const CycleFixture& fixture : cycleFixtures() )
    {
        INFO( fixture.name );
        const PlacedLayout layout = cycleLayout( fixture );
        CHECK( layout.connections.size() == 8 );
        size_t nontrivial = 0;
        for( const CycleInterface& i : fixture.interfaces )
        {
            const auto& [p, b] = layout.connections.at( ConstituentSide{ i.pa, i.sa } );
            CHECK( b.constituent_id == i.pb );
            CHECK( b.side_id == i.sb );
            CHECK( p == i.permutation );
            CHECK( sideClass( layout.patches.at( i.pa ), i.sa, layout.patches.at( i.pb ), i.sb ) ==
                   std::pair{ i.swapped, i.reversed } );
            if( i.swapped or i.reversed > 0 ) nontrivial++;
            if( fixture.name.starts_with( "3D" ) and not fixture.name.ends_with( "aligned" ) )
            {
                permutations.insert( i.permutation );
                const bool first = cycle_refined_patches.contains( i.pa );
                if( i.swapped ) owned.insert( { "swapped", first } );
                if( i.reversed > 0 ) owned.insert( { "reversed", first } );
            }
        }
        CHECK( nontrivial == ( fixture.name.ends_with( "aligned" ) ? 0 : 2 ) );
    }
    // Across the mixed 3D cycles: every permutation, and swapped and reversed
    // interfaces under each direction of coarse-fine ownership.
    CHECK( permutations.size() == 4 );
    const std::set<std::pair<std::string, bool>> both = {
        { "reversed", false }, { "reversed", true }, { "swapped", false }, { "swapped", true } };
    CHECK( owned == both );
}

TEST_CASE( "Placed four-patch cycles: junction classes, leaves, fragments and physical traces",
           "[shared_interface_refinement]" )
{
    std::vector<std::string> problems;
    for( const CycleFixture& fixture : cycleFixtures() )
    {
        INFO( fixture.name );
        const bool three_d = fixture.rotations.front().size() == 3;
        const PlacedLayout layout = cycleLayout( fixture );
        const CyclePins pins = cyclePins( layout );
        problems.insert( problems.end(), pins.problems.begin(), pins.problems.end() );
        if( pins.spans.size() != 3 ) continue;
        CHECK( pins.spans.at( "H1" ).at( 4 ) == ( three_d ? 4 : 1 ) );
        CHECK( not pins.spans.at( "H(div)" ).contains( 4 ) );
        CHECK( pins.spans.at( "H(curl)" ).contains( 4 ) == three_d );
        if( three_d ) CHECK( pins.spans.at( "H(curl)" ).at( 4 ) == 3 );

        // A throw from the geometry must not hide the pins above.
        PlacedOutcome o;
        try
        {
            o = placedLayoutOutcome( layout );
        }
        catch( const std::exception& e )
        {
            problems.push_back( fixture.name + ", placedLayoutOutcome threw: " + e.what() );
            continue;
        }
        CHECK( o.expected_records == layout.expected_records );
        problems.insert( problems.end(), o.problems.begin(), o.problems.end() );
        for( const auto& [name, t] : o.traces )
        {
            INFO( name );
            CHECK( t.records == layout.expected_records );
            CHECK( t.nonzero_shared_trace_observations > 0 );
            if( name != "H1" ) CHECK( t.full_vector_jump_observations > 0 );
            problems.insert( problems.end(), t.problems.begin(), t.problems.end() );
        }
    }
    INFO( joined( problems ) );
    CHECK( problems.empty() );
}

// Step 9: the discrete complex, sampled in each patch's parametric
// coordinates.  Piola maps commute with the differentials, so these are the
// physical relations patch by patch.  On each leaf every function and image
// is a polynomial of degree at most 2 per axis, which a 3^dim grid
// determines, so equal samples are equal functions.
namespace
{
    enum class Derivative
    {
        Grad,
        Curl,
        Rot,
        Div,
        None
    };

    // derivs: rows local functions, columns direction * ncomp + component.
    Eigen::MatrixXd
        applyDerivative( const Derivative d, const Eigen::MatrixXd& derivs, const size_t dim, const size_t ncomp )
    {
        const auto at = [&]( const size_t dir, const size_t comp ) { return derivs.col( dir * ncomp + comp ); };
        Eigen::MatrixXd out;
        switch( d )
        {
            case Derivative::Grad:
                out.resize( derivs.rows(), dim );
                for( size_t g = 0; g < dim; g++ ) out.col( g ) = at( g, 0 );
                break;
            case Derivative::Curl:
                if( ncomp == 1 )
                {
                    out.resize( derivs.rows(), 2 );
                    out.col( 0 ) = at( 1, 0 );
                    out.col( 1 ) = -at( 0, 0 );
                }
                else
                {
                    out.resize( derivs.rows(), 3 );
                    out.col( 0 ) = at( 1, 2 ) - at( 2, 1 );
                    out.col( 1 ) = at( 2, 0 ) - at( 0, 2 );
                    out.col( 2 ) = at( 0, 1 ) - at( 1, 0 );
                }
                break;
            case Derivative::Rot:
                out.resize( derivs.rows(), 1 );
                out.col( 0 ) = at( 0, 1 ) - at( 1, 0 );
                break;
            case Derivative::Div:
                out = Eigen::MatrixXd::Zero( derivs.rows(), 1 );
                for( size_t g = 0; g < dim; g++ ) out.col( 0 ) += at( g, g );
                break;
            case Derivative::None: break;
        }
        return out;
    }

    struct LeafSample
    {
        std::vector<FunctionId> conn;
        Eigen::MatrixXd values;      // local function, component
        Eigen::MatrixXd derivatives; // local function, component of the image
    };

    using SampleKey = std::vector<long>; // patch, then the parametric point in 1e-9 units

    // Throws if two leaves place a sample at the same patch point.
    std::map<SampleKey, LeafSample> sampleLeaves( const SplineSpace& ss, const Derivative d )
    {
        const auto& cmap =
            dynamic_cast<const HierarchicalMultiPatchCombinatorialMap&>( ss.basisComplex().parametricAtlas().cmap() );
        const size_t dim = cmap.dim();
        const std::array<double, 3> grid = { 0.13, 0.47, 0.86 };
        size_t n_points = 1;
        for( size_t k = 0; k < dim; k++ ) n_points *= grid.size();
        eval::SplineSpaceEvaluator evals( ss, 1 );
        std::map<SampleKey, LeafSample> out;
        for( const Cell& c : topCells( cmap ) )
        {
            const size_t patch = leafKeyOf( cmap, c ).at( 1 );
            evals.localizeElement( c );
            const std::vector<FunctionId> conn = ss.connectivity( c );
            for( size_t i = 0; i < n_points; i++ )
            {
                Eigen::VectorXd xi( dim );
                for( size_t k = 0, rest = i; k < dim; k++, rest /= grid.size() )
                    xi( k ) = grid.at( rest % grid.size() );
                evals.localizeParentPoint( cubePoint( xi ) );
                const Eigen::VectorXd p = evals.evaluateParametricPoint();
                SampleKey key{ static_cast<long>( patch ) };
                for( Eigen::Index k = 0; k < p.size(); k++ ) key.push_back( std::lround( p( k ) * 1e9 ) );
                LeafSample s{ conn, evals.evaluateBasisValuesAtParentPoint(), Eigen::MatrixXd() };
                if( d != Derivative::None )
                    s.derivatives = applyDerivative( d,
                                                     evals.evaluateBasisFirstDerivativesWrtParametricCoordinates(),
                                                     dim,
                                                     ss.numVectorComponents() );
                if( not out.emplace( key, s ).second ) throw std::runtime_error( "two leaves share a sample point" );
            }
        }
        return out;
    }

    enum class Trace
    {
        Value,
        Tangential,
        Normal,
        None
    };

    // Trace samples where leaf sides lie on the sides of the box [lo, hi]^dim:
    // rows sample and constrained component, columns functions.  Three points
    // per tangential direction determine a trace of degree at most 2 in each.
    Eigen::MatrixXd boundaryTraces( const SplineSpace& ss, const Trace trace, const double lo, const double hi )
    {
        const Eigen::Index nf = Eigen::Index( ss.numFunctions() );
        if( trace == Trace::None ) return Eigen::MatrixXd( 0, nf );
        const auto& cmap = ss.basisComplex().parametricAtlas().cmap();
        const size_t dim = cmap.dim();
        const std::array<double, 3> grid = { 0.13, 0.47, 0.86 };
        size_t n_points = 1;
        for( size_t k = 0; k + 1 < dim; k++ ) n_points *= grid.size();
        eval::SplineSpaceEvaluator evals( ss, 0 );
        std::vector<Eigen::RowVectorXd> rows;
        for( const Cell& c : topCells( cmap ) )
        {
            evals.localizeElement( c );
            const std::vector<FunctionId> conn = ss.connectivity( c );
            for( size_t side = 0; side < 2 * dim; side++ )
                for( size_t i = 0; i < n_points; i++ )
                {
                    Eigen::VectorXd xi( dim );
                    for( size_t k = 0, rest = i; k < dim; k++ )
                    {
                        if( k == side / 2 )
                            xi( k ) = double( side % 2 );
                        else
                        {
                            xi( k ) = grid.at( rest % grid.size() );
                            rest /= grid.size();
                        }
                    }
                    evals.localizeParentPoint( cubePoint( xi ) );
                    const Eigen::VectorXd p = evals.evaluateParametricPoint();
                    std::vector<size_t> normal;
                    for( size_t g = 0; g < dim; g++ )
                        if( std::abs( p( g ) - lo ) < geom_tol or std::abs( p( g ) - hi ) < geom_tol )
                            normal.push_back( g );
                    if( normal.empty() ) continue;
                    if( normal.size() > 1 ) throw std::runtime_error( "a side sample sits on an edge of the box" );
                    const Eigen::MatrixXd values = evals.evaluateBasisValuesAtParentPoint();
                    for( size_t comp = 0; comp < ss.numVectorComponents(); comp++ )
                    {
                        const bool tangential = comp != normal.front();
                        if( ( trace == Trace::Tangential and not tangential ) or
                            ( trace == Trace::Normal and tangential ) )
                            continue;
                        Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero( nf );
                        for( size_t j = 0; j < conn.size(); j++ )
                            row( Eigen::Index( conn.at( j ).id() ) ) =
                                values( Eigen::Index( j ), Eigen::Index( comp ) );
                        rows.push_back( row );
                    }
                }
        }
        Eigen::MatrixXd out( Eigen::Index( rows.size() ), nf );
        for( size_t r = 0; r < rows.size(); r++ ) out.row( Eigen::Index( r ) ) = rows.at( r );
        return out;
    }

    struct ChainSpace
    {
        std::string name;
        const SplineSpace* ss;
        Derivative d;
        Trace trace; // set to zero by homogeneous boundary conditions
    };

    struct ChainOutcome
    {
        std::vector<size_t> dims;
        std::vector<size_t> ranks; // of d_k; zero for the last space
        std::vector<long> betti;
        double inclusion = 0;    // fitting residual of d V_k in V_k+1, relative
        double composition = 0;  // d_k+1 d_k, relative
        double independence = 1; // smallest singular value of the samples, relative
        double kernel_edge = 0;  // largest singular value of a d_k counted zero, relative
        double image_edge = 1;   // smallest counted nonzero, relative
        std::vector<std::string> problems;
        // With homogeneous boundary conditions, if a box was given.
        std::vector<size_t> bc_dims;
        std::vector<size_t> bc_ranks;
        std::vector<long> bc_betti;
        double bc_inclusion = 0; // trace of d of a zero-trace function, relative
    };

    constexpr double rank_tol = 1e-8;

    // std::max and std::min return their first argument if the second is NaN.
    // Metrics accumulate through these instead: a NaN or an infinity becomes
    // NaN and stays.
    constexpr double nonfinite = std::numeric_limits<double>::quiet_NaN();

    double maxOrNaN( const double a, const double b )
    {
        return std::isfinite( a ) and std::isfinite( b ) ? std::max( a, b ) : nonfinite;
    }

    double minOrNaN( const double a, const double b )
    {
        return std::isfinite( a ) and std::isfinite( b ) ? std::min( a, b ) : nonfinite;
    }

    // Zero if empty; NaN if an entry is not finite.
    double maxAbsOrNaN( const Eigen::MatrixXd& m )
    {
        if( not m.allFinite() ) return nonfinite;
        return m.size() == 0 ? 0 : m.cwiseAbs().maxCoeff();
    }

    // max|D_k+1 D_k| / (max|D_k+1| max|D_k|).  Zero over zero is zero: a zero
    // differential is valid.  NaN if an entry is nonfinite, or if a nonzero
    // numerator meets a zero denominator.
    double compositionResidual( const Eigen::MatrixXd& next, const Eigen::MatrixXd& d )
    {
        const double numerator = maxAbsOrNaN( next * d );
        const double denominator = maxAbsOrNaN( next ) * maxAbsOrNaN( d );
        if( not std::isfinite( numerator ) or not std::isfinite( denominator ) ) return nonfinite;
        if( denominator == 0 ) return numerator == 0 ? 0 : nonfinite;
        return numerator / denominator;
    }

    using ChainSamples = std::vector<std::map<SampleKey, LeafSample>>;

    // A box adds homogeneous boundary conditions on its sides.  A perturbation
    // alters the samples first (the nonfinite regression).
    ChainOutcome chainOutcome( const std::string& prefix,
                               const std::vector<ChainSpace>& chain,
                               const std::optional<std::array<double, 2>>& box = std::nullopt,
                               const std::function<void( ChainSamples& )>& perturb = {} )
    {
        ChainOutcome out;
        const size_t n = chain.size();
        ChainSamples samples;
        for( const ChainSpace& c : chain ) samples.push_back( sampleLeaves( *c.ss, c.d ) );
        if( perturb ) perturb( samples );
        std::map<SampleKey, size_t> index;
        for( const auto& [key, s] : samples.front() ) index.emplace( key, index.size() );
        for( size_t k = 1; k < n; k++ )
        {
            bool same = samples.at( k ).size() == index.size();
            for( const auto& [key, s] : samples.at( k ) ) same = same and index.contains( key );
            if( not same ) out.problems.push_back( prefix + chain.at( k ).name + " is sampled at other points" );
        }
        if( not out.problems.empty() ) return out;

        // Sparse samples.  A Gram matrix is positive definite iff its samples
        // are injective; it fits d V_k in V_k+1 by normal equations.
        std::vector<Eigen::SparseMatrix<double>> values( n ), images( n );
        std::vector<Eigen::LDLT<Eigen::MatrixXd>> grams;
        for( size_t k = 0; k < n; k++ )
        {
            const size_t nc = chain.at( k ).ss->numVectorComponents();
            const size_t nf = chain.at( k ).ss->numFunctions();
            const size_t nc_next = k + 1 < n ? chain.at( k + 1 ).ss->numVectorComponents() : 0;
            out.dims.push_back( nf );
            std::vector<Eigen::Triplet<double>> v, im;
            for( const auto& [key, s] : samples.at( k ) )
            {
                const size_t row = index.at( key );
                if( chain.at( k ).d != Derivative::None and size_t( s.derivatives.cols() ) != nc_next )
                    throw std::runtime_error( prefix + chain.at( k ).name + ": the image has the wrong components" );
                for( size_t j = 0; j < s.conn.size(); j++ )
                {
                    const int f = static_cast<int>( s.conn.at( j ).id() );
                    for( size_t c = 0; c < nc; c++ ) v.emplace_back( int( row * nc + c ), f, s.values( j, c ) );
                    for( size_t c = 0; c < nc_next and chain.at( k ).d != Derivative::None; c++ )
                        im.emplace_back( int( row * nc_next + c ), f, s.derivatives( j, c ) );
                }
            }
            values.at( k ).resize( int( index.size() * nc ), int( nf ) );
            values.at( k ).setFromTriplets( v.begin(), v.end() );
            images.at( k ).resize( int( index.size() * nc_next ), int( nf ) );
            images.at( k ).setFromTriplets( im.begin(), im.end() );
            const Eigen::MatrixXd gram = Eigen::MatrixXd( values.at( k ).transpose() * values.at( k ) );
            double ratio = nonfinite;
            if( gram.allFinite() )
            {
                const Eigen::VectorXd ev =
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>( gram, Eigen::EigenvaluesOnly ).eigenvalues();
                ratio = std::sqrt( maxOrNaN( ev( 0 ), 0.0 ) / ev( ev.size() - 1 ) );
            }
            out.independence = minOrNaN( out.independence, ratio );
            grams.emplace_back( gram );
        }

        // Rank relative to the largest singular value; an empty or zero
        // spectrum has rank zero.  A nonfinite matrix is not decomposed: both
        // edges become NaN.  A null space, if asked for, is orthonormal (NaN if
        // the matrix is nonfinite).
        const auto countRank = [&]( const Eigen::MatrixXd& m, Eigen::MatrixXd* null_space = nullptr ) {
            size_t rank = 0;
            if( not m.allFinite() )
            {
                out.kernel_edge = out.image_edge = nonfinite;
                if( null_space ) *null_space = Eigen::MatrixXd::Constant( m.cols(), m.cols(), nonfinite );
                return rank;
            }
            const Eigen::BDCSVD<Eigen::MatrixXd> svd( m, null_space ? unsigned( Eigen::ComputeFullV ) : 0u );
            const Eigen::VectorXd& sv = svd.singularValues();
            if( sv.size() > 0 and sv( 0 ) != 0 )
                for( Eigen::Index i = 0; i < sv.size(); i++ )
                {
                    const double rel = sv( i ) / sv( 0 );
                    if( rel > rank_tol )
                    {
                        rank++;
                        out.image_edge = minOrNaN( out.image_edge, rel );
                    }
                    else
                        out.kernel_edge = maxOrNaN( out.kernel_edge, rel );
                }
            if( null_space ) *null_space = svd.matrixV().rightCols( m.cols() - Eigen::Index( rank ) );
            return rank;
        };
        std::vector<Eigen::MatrixXd> diffs;
        for( size_t k = 0; k + 1 < n; k++ )
        {
            const Eigen::MatrixXd D =
                grams.at( k + 1 ).solve( Eigen::MatrixXd( values.at( k + 1 ).transpose() * images.at( k ) ) );
            const Eigen::MatrixXd image = Eigen::MatrixXd( images.at( k ) );
            const double scale = maxOrNaN( maxAbsOrNaN( image ), 1.0 );
            out.inclusion = maxOrNaN( out.inclusion, maxAbsOrNaN( values.at( k + 1 ) * D - image ) / scale );
            out.ranks.push_back( countRank( D ) );
            diffs.push_back( D );
        }
        out.ranks.push_back( 0 );
        for( size_t k = 0; k + 1 < diffs.size(); k++ )
            out.composition = maxOrNaN( out.composition, compositionResidual( diffs.at( k + 1 ), diffs.at( k ) ) );
        for( size_t k = 0; k < n; k++ )
            out.betti.push_back( long( out.dims.at( k ) ) - long( out.ranks.at( k ) ) -
                                 ( k > 0 ? long( out.ranks.at( k - 1 ) ) : 0 ) );
        if( not box ) return out;

        // Zero-trace subspaces: orthonormal null spaces of the trace samples.
        std::vector<Eigen::MatrixXd> traces, zero_traces;
        for( size_t k = 0; k < n; k++ )
        {
            traces.push_back( boundaryTraces( *chain.at( k ).ss, chain.at( k ).trace, box->at( 0 ), box->at( 1 ) ) );
            const Eigen::Index nf = Eigen::Index( out.dims.at( k ) );
            if( traces.back().rows() == 0 )
                zero_traces.push_back( Eigen::MatrixXd::Identity( nf, nf ) );
            else
            {
                Eigen::MatrixXd z;
                countRank( traces.back(), &z );
                zero_traces.push_back( z );
            }
            out.bc_dims.push_back( size_t( zero_traces.back().cols() ) );
        }
        for( size_t k = 0; k + 1 < n; k++ )
        {
            const Eigen::MatrixXd dz = diffs.at( k ) * zero_traces.at( k );
            const Eigen::MatrixXd image = images.at( k ) * zero_traces.at( k );
            if( traces.at( k + 1 ).rows() > 0 and dz.cols() > 0 )
                out.bc_inclusion = maxOrNaN( out.bc_inclusion,
                                             maxAbsOrNaN( traces.at( k + 1 ) * dz ) /
                                                 maxOrNaN( maxAbsOrNaN( image ), 1.0 ) );
            out.bc_ranks.push_back( countRank( dz ) );
        }
        out.bc_ranks.push_back( 0 );
        for( size_t k = 0; k < n; k++ )
            out.bc_betti.push_back( long( out.bc_dims.at( k ) ) - long( out.bc_ranks.at( k ) ) -
                                    ( k > 0 ? long( out.bc_ranks.at( k - 1 ) ) : 0 ) );
        return out;
    }

    // One chain in 3D, two in 2D.  The vector spaces come from glued_h1; h1
    // starts each chain.  A box adds homogeneous boundary conditions; a
    // perturbation goes to chainOutcome.
    std::vector<std::pair<std::string, ChainOutcome>>
        complexOutcomes( const std::string& name,
                         const HierarchicalMultiPatchSplineSpace& glued_h1,
                         const HierarchicalMultiPatchSplineSpace& h1,
                         const std::optional<std::array<double, 2>>& box = std::nullopt,
                         const std::function<void( ChainSamples& )>& perturb = {} )
    {
        const size_t dim = h1.basisComplex().parametricAtlas().cmap().dim();
        const VectorConformingHierarchicalMultiPatchSplineSpace hcurl =
            buildHCurlHierarchicalMultiPatchSplineSpace( glued_h1 );
        const VectorConformingHierarchicalMultiPatchSplineSpace hdiv =
            buildHDivHierarchicalMultiPatchSplineSpace( glued_h1 );
        const HierarchicalMultiPatchSplineSpace l2 = buildL2HierarchicalMultiPatchSplineSpace( hdiv );
        std::vector<std::pair<std::string, std::vector<ChainSpace>>> chains;
        if( dim == 2 )
            chains = { { "grad, rot",
                         { { "H1", &h1, Derivative::Grad, Trace::Value },
                           { "H(curl)", &hcurl, Derivative::Rot, Trace::Tangential },
                           { "L2", &l2, Derivative::None, Trace::None } } },
                       { "curl, div",
                         { { "H1", &h1, Derivative::Curl, Trace::Value },
                           { "H(div)", &hdiv, Derivative::Div, Trace::Normal },
                           { "L2", &l2, Derivative::None, Trace::None } } } };
        else
            chains = { { "grad, curl, div",
                         { { "H1", &h1, Derivative::Grad, Trace::Value },
                           { "H(curl)", &hcurl, Derivative::Curl, Trace::Tangential },
                           { "H(div)", &hdiv, Derivative::Div, Trace::Normal },
                           { "L2", &l2, Derivative::None, Trace::None } } } };
        std::vector<std::pair<std::string, ChainOutcome>> out;
        for( const auto& [chain_name, chain] : chains )
            out.emplace_back( chain_name, chainOutcome( name + ", " + chain_name + ": ", chain, box, perturb ) );
        return out;
    }

    std::vector<std::pair<std::string, ChainOutcome>> complexOutcomes( const PlacedLayout& layout )
    {
        const LayoutHierarchy hierarchy = placedHierarchy( layout, placedLeaves( layout ) );
        const HierarchicalMultiPatchSplineSpace h1 = buildHierarchicalSplineSpace( hierarchy.levels, hierarchy.leaves );
        return complexOutcomes( layout.name, h1, h1 );
    }

    // H1 from the same patches and leaves with no interfaces starts each chain.
    std::vector<std::pair<std::string, ChainOutcome>> ungluedH1Outcomes( const PlacedLayout& layout )
    {
        const LayoutHierarchy hierarchy = placedHierarchy( layout, placedLeaves( layout ) );
        const HierarchicalMultiPatchSplineSpace h1 = buildHierarchicalSplineSpace( hierarchy.levels, hierarchy.leaves );
        PlacedLayout apart = layout;
        apart.connections.clear();
        const LayoutHierarchy apart_hierarchy = placedHierarchy( apart, placedLeaves( apart ) );
        const HierarchicalMultiPatchSplineSpace unglued =
            buildHierarchicalSplineSpace( apart_hierarchy.levels, apart_hierarchy.leaves );
        return complexOutcomes( layout.name, h1, unglued );
    }

    // Glues every pair of patches whose shifts differ by one along an axis.
    void glueTouching( PlacedLayout& layout )
    {
        const size_t dim = layout.patches.front().rotation.size();
        const auto facing = [&]( const size_t p, const size_t g, const int sign ) {
            for( size_t s = 0; s < 2 * dim; s++ )
                if( placedNormal( layout.patches.at( p ).rotation, s ) == std::pair<size_t, int>{ g, sign } ) return s;
            throw std::runtime_error( "no side faces that way" );
        };
        for( size_t a = 0; a < layout.patches.size(); a++ )
            for( size_t b = a + 1; b < layout.patches.size(); b++ )
                for( size_t g = 0; g < dim; g++ )
                {
                    const Eigen::VectorXd step = layout.patches.at( b ).shift - layout.patches.at( a ).shift;
                    const Eigen::VectorXd e = Eigen::VectorXd::Unit( dim, g );
                    if( ( step - e ).norm() < geom_tol ) glue( layout, a, facing( a, g, 1 ), b, facing( b, g, -1 ) );
                    if( ( step + e ).norm() < geom_tol ) glue( layout, a, facing( a, g, -1 ), b, facing( b, g, 1 ) );
                }
    }

    // Eight unit cubes around a hole: an annulus in 2D, a solid torus in 3D.
    // Refined, patch 0 refines the elements facing patch 1.
    PlacedLayout ringLayout( const size_t dim, const bool refined )
    {
        PlacedLayout layout;
        layout.name = std::to_string( dim ) + "D ring of eight" + ( refined ? ", refined" : ", unrefined" );
        Rotation identity( dim, std::vector<int>( dim, 0 ) );
        for( size_t k = 0; k < dim; k++ ) identity.at( k ).at( k ) = 1;
        const std::vector<std::pair<int, int>> cells = { { 0, 0 }, { 1, 0 }, { 2, 0 }, { 2, 1 },
                                                         { 2, 2 }, { 1, 2 }, { 0, 2 }, { 0, 1 } };
        for( const auto& [x, y] : cells )
        {
            Eigen::VectorXd shift = Eigen::VectorXd::Zero( dim );
            shift( 0 ) = x;
            shift( 1 ) = y;
            layout.patches.push_back( { identity, shift } );
        }
        glueTouching( layout );
        layout.refined_axes = { 0, 1 };
        if( refined )
            for( size_t e = 0; e < ( size_t( 1 ) << ( dim - 1 ) ); e++ )
            {
                std::vector<size_t> element{ 1 };
                for( size_t k = 0; k + 1 < dim; k++ ) element.push_back( ( e >> k ) & 1 );
                layout.refined.push_back( { 0, element } );
            }
        return layout;
    }

    PlacedLayout unrefined( PlacedLayout layout )
    {
        layout.name += ", unrefined";
        layout.refined.clear();
        return layout;
    }

    // Every 2D two-patch layout, the first 3D layout of each class, the cycles.
    std::vector<PlacedLayout> contractibleLayouts()
    {
        std::vector<PlacedLayout> out = twoPatchPlacedLayouts( 2 );
        std::set<std::string> classes;
        for( const PlacedLayout& layout : twoPatchPlacedLayouts( 3 ) )
            if( classes.insert( layout.orientation ).second ) out.push_back( layout );
        for( const CycleFixture& fixture : cycleFixtures() ) out.push_back( cycleLayout( fixture ) );
        return out;
    }

    // Shepherd and Toshniwal, Fig. 7(b) (ledger, step 9): one biquadratic
    // patch of 7 x 7 elements on [0,7]^2; refined, the elements in [1,4]^2 and
    // [3,6]^2 are bisected.
    HierarchicalMultiPatchSplineSpace twoSquaresH1( const bool refined )
    {
        const double ptol = 1e-10;
        std::vector<std::shared_ptr<const MultiPatchSplineSpace>> levels;
        for( size_t level = 0; level < 2; level++ )
        {
            std::vector<double> knots{ 0, 0 };
            for( size_t i = 0; i <= ( size_t( 7 ) << level ); i++ )
                knots.push_back( double( i ) / double( size_t( 1 ) << level ) );
            knots.insert( knots.end(), { 7, 7 } );
            const KnotVector kv( knots, ptol );
            levels.push_back( std::make_shared<const MultiPatchSplineSpace>(
                buildH1MultiPatchSplineSpace( { makePatch( { kv, kv }, { 2, 2 } ) }, {} ) ) );
        }
        const auto inSquare = []( const size_t s, const size_t t, const size_t corner ) {
            return corner <= s and s < corner + 3 and corner <= t and t < corner + 3;
        };
        std::vector<std::vector<Cell>> leaves( 2 );
        for( size_t s = 0; s < 7; s++ )
            for( size_t t = 0; t < 7; t++ )
            {
                if( refined and ( inSquare( s, t, 1 ) or inSquare( s, t, 3 ) ) )
                {
                    for( size_t i = 0; i < 4; i++ )
                        leaves.at( 1 ).push_back(
                            patchTopCellAt( levelCMap( *levels.at( 1 ) ), 0, { 2 * s + i % 2, 2 * t + i / 2 } ) );
                }
                else
                    leaves.at( 0 ).push_back( patchTopCellAt( levelCMap( *levels.at( 0 ) ), 0, { s, t } ) );
            }
        return buildHierarchicalSplineSpace( levels, leaves );
    }

    // Tolerances sit far from the measured worst cases (outstanding_issues.txt,
    // step 9); a NaN or an infinity fails each one.  Betti numbers are reported
    // per degree; a nonempty bc_betti also checks the zero-trace chain.
    std::vector<std::string> complexProblems( const std::string& where,
                                              const ChainOutcome& o,
                                              const std::vector<long>& betti,
                                              const std::vector<long>& bc_betti = {} )
    {
        if( not o.problems.empty() ) return o.problems;
        const auto below = []( const double x, const double tol ) { return std::isfinite( x ) and x < tol; };
        const auto above = []( const double x, const double tol ) { return std::isfinite( x ) and x > tol; };
        std::vector<std::string> out;
        if( not below( o.inclusion, 1e-12 ) ) out.push_back( where + "d V_k is not in V_k+1" );
        if( not below( o.composition, 1e-12 ) ) out.push_back( where + "d_k+1 d_k is not zero" );
        if( not above( o.independence, 1e-3 ) ) out.push_back( where + "a basis is not linearly independent" );
        if( not( below( o.kernel_edge, 1e-12 ) and above( o.image_edge, 1e-3 ) ) )
            out.push_back( where + "a rank is not clear-cut" );
        const auto perDegree = [&]( const std::string& label,
                                    const std::vector<long>& got,
                                    const std::vector<long>& expected ) {
            if( got.size() != expected.size() )
                out.push_back( where + label + std::to_string( got.size() ) + " Betti numbers, expected " +
                               std::to_string( expected.size() ) );
            for( size_t k = 0; k < std::min( got.size(), expected.size() ); k++ )
                if( got.at( k ) != expected.at( k ) )
                    out.push_back( where + label + "b_" + std::to_string( k ) + " = " + std::to_string( got.at( k ) ) +
                                   ", expected " + std::to_string( expected.at( k ) ) );
        };
        perDegree( "", o.betti, betti );
        if( bc_betti.empty() ) return out;
        if( not below( o.bc_inclusion, 1e-12 ) ) out.push_back( where + "d of a zero-trace function has a trace" );
        perDegree( "zero traces, ", o.bc_betti, bc_betti );
        return out;
    }

    // Expects 1 in degree 0, loops in degree 1, zero above.  A throw is
    // reported per layout.
    std::vector<std::string> complexGate( const std::vector<PlacedLayout>& layouts, const size_t loops, size_t& chains )
    {
        std::vector<std::string> out;
        for( const PlacedLayout& layout : layouts )
        {
            std::vector<long> betti( layout.patches.front().rotation.size() + 1, 0 );
            betti.at( 0 ) = 1;
            betti.at( 1 ) = long( loops );
            try
            {
                for( const auto& [name, o] : complexOutcomes( layout ) )
                {
                    const std::vector<std::string> p = complexProblems( layout.name + ", " + name + ": ", o, betti );
                    out.insert( out.end(), p.begin(), p.end() );
                    chains++;
                }
            }
            catch( const std::exception& e )
            {
                out.push_back( layout.name + ": " + e.what() );
            }
        }
        return out;
    }
}

TEST_CASE( "Discrete complex on contractible layouts without refinement: inclusions, composition and exactness",
           "[shared_interface_refinement]" )
{
    std::vector<PlacedLayout> layouts;
    for( const PlacedLayout& layout : contractibleLayouts() ) layouts.push_back( unrefined( layout ) );
    CHECK( layouts.size() == 16 + 10 + 5 );
    size_t chains = 0;
    const std::vector<std::string> problems = complexGate( layouts, 0, chains );
    CHECK( chains == 2 * 16 + 10 + 2 * 2 + 3 );
    INFO( joined( problems ) );
    CHECK( problems.empty() );
}

TEST_CASE( "Discrete complex on contractible layouts under asymmetric leaves: inclusions, composition and exactness",
           "[shared_interface_refinement]" )
{
    const std::vector<PlacedLayout> layouts = contractibleLayouts();
    CHECK( layouts.size() == 16 + 10 + 5 );
    size_t chains = 0;
    const std::vector<std::string> problems = complexGate( layouts, 0, chains );
    CHECK( chains == 2 * 16 + 10 + 2 * 2 + 3 );
    INFO( joined( problems ) );
    CHECK( problems.empty() );
}

TEST_CASE( "Discrete complex on a ring of eight patches: one loop, with and without refinement",
           "[shared_interface_refinement]" )
{
    const std::vector<PlacedLayout> layouts = {
        ringLayout( 2, false ), ringLayout( 2, true ), ringLayout( 3, false ), ringLayout( 3, true ) };
    for( const PlacedLayout& layout : layouts ) CHECK( layout.connections.size() == 2 * 8 );
    size_t chains = 0;
    const std::vector<std::string> problems = complexGate( layouts, 1, chains );
    CHECK( chains == 2 * 2 + 2 );
    INFO( joined( problems ) );
    CHECK( problems.empty() );
}

TEST_CASE( "Discrete complex: the inclusion check sees an unglued interface", "[shared_interface_refinement]" )
{
    // Grad (2D: also curl) of H1 built without interfaces leaves the glued
    // space at the interfaces.
    const std::vector<PlacedLayout> two_d = twoPatchPlacedLayouts( 2 );
    const std::vector<PlacedLayout> three_d = twoPatchPlacedLayouts( 3 );
    std::vector<PlacedLayout> layouts = {
        unrefined( two_d.front() ), two_d.back(), unrefined( three_d.front() ), three_d.back() };
    for( const CycleFixture& fixture : cycleFixtures() ) layouts.push_back( cycleLayout( fixture ) );
    std::vector<std::string> problems;
    size_t chains = 0;
    for( const PlacedLayout& layout : layouts )
    {
        try
        {
            for( const auto& [name, o] : ungluedH1Outcomes( layout ) )
            {
                const std::string where = layout.name + ", " + name + ": ";
                problems.insert( problems.end(), o.problems.begin(), o.problems.end() );
                if( not( std::isfinite( o.inclusion ) and o.inclusion > 0.1 ) )
                    problems.push_back( where + "the unglued gradients fit" );
                chains++;
            }
        }
        catch( const std::exception& e )
        {
            problems.push_back( layout.name + ": " + e.what() );
        }
    }
    CHECK( chains == 2 + 2 + 1 + 1 + 2 * 2 + 3 );
    INFO( joined( problems ) );
    CHECK( problems.empty() );
}

TEST_CASE( "Discrete complex on a published non-exact hierarchy: Betti numbers per degree",
           "[shared_interface_refinement]" )
{
    // Shepherd and Toshniwal, Fig. 7(b) (ledger, step 9).  With homogeneous
    // boundary conditions exact is 0 0 1, and the refinement adds one harmonic
    // 1-form.  Without them the ledger's trace sequence gives 1 1 0.
    // Dimensions are counted by hand from the selection rule.
    struct Expected
    {
        bool refined;
        std::vector<size_t> dims, bc_dims;
        std::vector<long> betti, bc_betti;
    };
    const std::vector<Expected> fixtures = { { false, { 81, 144, 64 }, { 49, 112, 64 }, { 1, 0, 0 }, { 0, 0, 1 } },
                                             { true, { 111, 216, 105 }, { 79, 184, 105 }, { 1, 1, 0 }, { 0, 1, 1 } } };
    std::vector<std::string> problems;
    std::vector<long> harmonic_one_forms; // refined, zero traces, per chain
    size_t chains = 0;
    for( const Expected& e : fixtures )
    {
        const std::string name = e.refined ? "two squares" : "two squares, unrefined";
        try
        {
            const HierarchicalMultiPatchSplineSpace h1 = twoSquaresH1( e.refined );
            if( topCells( h1.basisComplex().parametricAtlas().cmap() ).size() != ( e.refined ? 32 + 4 * 17 : 49 ) )
                problems.push_back( name + ": leaf count" );
            for( const auto& [chain_name, o] : complexOutcomes( name, h1, h1, std::array<double, 2>{ 0, 7 } ) )
            {
                const std::string where = name + ", " + chain_name + ": ";
                const std::vector<std::string> p = complexProblems( where, o, e.betti, e.bc_betti );
                problems.insert( problems.end(), p.begin(), p.end() );
                if( o.dims != e.dims or o.bc_dims != e.bc_dims ) problems.push_back( where + "dimensions" );
                if( e.refined )
                {
                    // The contractible expectation fails in degree 1 alone.
                    const std::vector<std::string> exact = complexProblems( where, o, { 1, 0, 0 }, { 0, 0, 1 } );
                    if( exact != std::vector<std::string>{ where + "b_1 = 1, expected 0",
                                                           where + "zero traces, b_1 = 1, expected 0" } )
                        problems.push_back( where + "the contractible expectation fails elsewhere: " +
                                            joined( exact ) );
                    if( o.bc_betti.size() == 3 ) harmonic_one_forms.push_back( o.bc_betti.at( 1 ) );
                }
                chains++;
            }
        }
        catch( const std::exception& ex )
        {
            problems.push_back( name + ": " + ex.what() );
        }
    }
    CHECK( chains == 2 * 2 );
    CHECK( harmonic_one_forms == std::vector<long>{ 1, 1 } );
    INFO( joined( problems ) );
    CHECK( problems.empty() );
}

TEST_CASE( "Discrete complex: a zero differential has composition residual zero", "[shared_interface_refinement]" )
{
    // Direct, on 2 x 2 matrices.  The nonzero ratio and the infinity are
    // controls: zero over zero must not make every residual zero, nor hide a
    // nonfinite entry.
    const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero( 2, 2 );
    const Eigen::MatrixXd d{ { 1, 2 }, { 3, 4 } };
    Eigen::MatrixXd infinite = zero;
    infinite( 1, 0 ) = std::numeric_limits<double>::infinity();
    CHECK( compositionResidual( d, zero ) == 0 );
    CHECK( compositionResidual( zero, d ) == 0 );
    CHECK( compositionResidual( zero, zero ) == 0 );
    CHECK( compositionResidual( d, d ) == 22.0 / 16.0 );
    CHECK( std::isnan( compositionResidual( zero, infinite ) ) );
}

TEST_CASE( "Discrete complex: a nonfinite sample reaches the problem report", "[shared_interface_refinement]" )
{
    // The published case, unrefined (one biquadratic patch of 7 x 7 elements
    // on [0,7]^2), with one sample set nonfinite in each chain.  A NaN in an
    // H1 derivative reaches d_0 and, through it, the zero-trace inclusion; an
    // infinity in an L2 value reaches the L2 Gram matrix and d_1.  One problem
    // reports both rank edges, so each edge is also checked.
    struct Fixture
    {
        std::string name;
        std::function<void( ChainSamples& )> perturb;
        std::vector<std::string> metrics; // problems expected, in report order
    };
    const std::vector<Fixture> fixtures = {
        { "control", {}, {} },
        { "NaN in an H1 derivative",
          []( ChainSamples& s ) {
              s.front().begin()->second.derivatives( 0, 0 ) = std::numeric_limits<double>::quiet_NaN();
          },
          { "d V_k is not in V_k+1",
            "d_k+1 d_k is not zero",
            "a rank is not clear-cut",
            "d of a zero-trace function has a trace" } },
        { "infinity in an L2 value",
          []( ChainSamples& s ) { s.back().begin()->second.values( 0, 0 ) = std::numeric_limits<double>::infinity(); },
          { "d V_k is not in V_k+1",
            "d_k+1 d_k is not zero",
            "a basis is not linearly independent",
            "a rank is not clear-cut" } } };
    const std::vector<std::string> all_metrics = { "d V_k is not in V_k+1",
                                                   "d_k+1 d_k is not zero",
                                                   "a basis is not linearly independent",
                                                   "a rank is not clear-cut",
                                                   "d of a zero-trace function has a trace" };
    std::vector<std::string> problems;
    size_t chains = 0;
    for( const Fixture& f : fixtures )
    {
        try
        {
            const HierarchicalMultiPatchSplineSpace h1 = twoSquaresH1( false );
            for( const auto& [chain_name, o] :
                 complexOutcomes( f.name, h1, h1, std::array<double, 2>{ 0, 7 }, f.perturb ) )
            {
                const std::string where = f.name + ", " + chain_name + ": ";
                const std::vector<std::string> p = complexProblems( where, o, { 1, 0, 0 }, { 0, 0, 1 } );
                if( f.metrics.empty() and not p.empty() ) problems.push_back( where + "unperturbed: " + joined( p ) );
                if( not f.metrics.empty() )
                {
                    std::vector<std::string> got;
                    for( const std::string& m : all_metrics )
                        if( std::find( p.begin(), p.end(), where + m ) != p.end() ) got.push_back( m );
                    if( got != f.metrics )
                        problems.push_back( where + "metric problems: " + joined( got ) + "; expected " +
                                            joined( f.metrics ) );
                    if( std::isfinite( o.kernel_edge ) ) problems.push_back( where + "the kernel edge is finite" );
                    if( std::isfinite( o.image_edge ) ) problems.push_back( where + "the image edge is finite" );
                }
                chains++;
            }
        }
        catch( const std::exception& e )
        {
            problems.push_back( f.name + ": " + e.what() );
        }
    }
    CHECK( chains == 3 * 2 );
    INFO( joined( problems ) );
    CHECK( problems.empty() );
}
