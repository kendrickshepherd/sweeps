// Item 1, step 4 (outstanding_issues.txt): algebraic correctness of asymmetric
// THB leaf selection across a same-mesh multipatch interface.
#include <catch2/catch_test_macros.hpp>
#include <HierarchicalMultiPatchSplineSpace.hpp>
#include <HierarchicalMultiPatchCombinatorialMap.hpp>
#include <KnotVector.hpp>
#include <MultiPatchSplineFactory.hpp>
#include <TraceExtraction.hpp>
#include <VectorConformingHierarchicalMultiPatchSplineSpace.hpp>
#include <CombinatorialMapMethods.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <set>
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

    // Two levels, patch 1 entirely coarse.  Checks: (1) coarse shared functions
    // active, (2) fine shared functions inactive, (3) truncation on fine
    // leaves, (4) rows on coarse leaves, (5) per-component partition of unity.
    void checkSharedInterfaceRefinement( const HierarchyView& v, const Fixture& fx )
    {
        INFO( v.name );
        REQUIRE( v.levels.size() == 2 );
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

        // The requested leaves tile each patch.
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
                    if( leaves.at( b ).patch == patch ) REQUIRE( overlap( leaves.at( a ).box, leaves.at( b ).box ) < geom_tol );
            }
            REQUIRE( std::abs( covered - patch_measure ) < geom_tol );
        }

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
