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
#include <CombinatorialMapMethods.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
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
