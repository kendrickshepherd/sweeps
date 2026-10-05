// Phase 0 characterization harness.  See outstanding_issues.txt items 1 and 2.
//
// Assertions are of two kinds and are labelled individually:
//   PERMANENT - invariants that hold independently of the interface
//               correspondence convention.
//   TEMPORARY - pins current behaviour still governed by an open defect, so a
//               silent semantic change fails loudly.  Each cites the issue
//               that retires it.
//
// Updated for the Stage 2 side-aware transform.  phi and both function-id
// merges were always correct; the defect was the side-blind trace and
// validator consumers, now fixed.  Sides 0<->1 in 2d are NOT reversed -- F_a
// cancels Flip1d -- so the correspondence in these fixtures is the identity.
//
// 3d scope limit: the 3d fixtures carry one interface element per patch, so no
// reversal or axis swap can change which element pairs with which.  They
// observe extraction column order only, never pairing.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <HierarchicalMultiPatchSplineSpace.hpp>
#include <HierarchicalMultiPatchCombinatorialMap.hpp>
#include <KnotVector.hpp>
#include <MultiPatchSplineFactory.hpp>
#include <TraceMesh.hpp>
#include <VectorConformingHierarchicalMultiPatchSplineSpace.hpp>
#include <VectorConformingMultiPatchSplineSpace.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>

using namespace basis;
using namespace topology;

namespace
{
    using InternalConnectionsMap = MultiPatchCombinatorialMap::InternalConnectionsMap;
    using ConstituentSide = MultiPatchCombinatorialMap::ConstituentSide;
    using TPPermutation = MultiPatchCombinatorialMap::TPPermutation;

    constexpr double geom_tol = 1e-12;
    constexpr double coeff_tol = 1e-10;

    Cell topCellAt( const TPCombinatorialMap& cmap, const std::vector<size_t>& element_indices )
    {
        FullyUnflattenedDart unflat;
        for( const size_t element_index : element_indices )
            unflat.unflat_darts.push_back( Dart( element_index ) );
        unflat.dart_pos = SmallVector<TPCombinatorialMap::TPDartPos, 2>(
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

    const char* permName( const TPPermutation p )
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

    // Observations for one paired interface element.  Returns data rather than
    // printing, so the tests can assert on it.
    struct PairObs
    {
        Eigen::Index a_rows = 0;
        Eigen::Index a_cols = 0;
        Eigen::Index b_rows = 0;
        Eigen::Index b_cols = 0;
        size_t dart_a = 0;
        size_t dart_b = 0;
        size_t shared = 0;
        size_t identity_col_match = 0;
        size_t reversed_col_match = 0;
    };

    std::vector<PairObs> observeInterfaces( const SplineSpace& ss )
    {
        std::vector<PairObs> out;
        for( const TraceMeshInterface& iface : patchTraceMeshInterfaces( ss ) )
        {
            const TraceInterfaceElement tr = traceInterfaceElement( ss, iface );
            REQUIRE( tr.second.has_value() ); // PERMANENT: a patch interface has two sides.
            const TraceSideData& A = tr.first;
            const TraceSideData& B = *tr.second;

            std::set<FunctionId> sa( A.connectivity.begin(), A.connectivity.end() );
            std::set<FunctionId> sb( B.connectivity.begin(), B.connectivity.end() );
            std::vector<FunctionId> shared;
            std::set_intersection(
                sa.begin(), sa.end(), sb.begin(), sb.end(), std::back_inserter( shared ) );

            PairObs o;
            o.a_rows = A.extraction.rows();
            o.a_cols = A.extraction.cols();
            o.b_rows = B.extraction.rows();
            o.b_cols = B.extraction.cols();
            o.dart_a = A.element.dart().id();
            o.dart_b = B.element.dart().id();
            o.shared = shared.size();

            for( const FunctionId& f : shared )
            {
                if( A.extraction.cols() != B.extraction.cols() ) continue;
                const Eigen::Index ia = static_cast<Eigen::Index>(
                    std::find( A.connectivity.begin(), A.connectivity.end(), f ) - A.connectivity.begin() );
                const Eigen::Index ib = static_cast<Eigen::Index>(
                    std::find( B.connectivity.begin(), B.connectivity.end(), f ) - B.connectivity.begin() );
                const Eigen::VectorXd ra = A.extraction.row( ia ).transpose();
                const Eigen::VectorXd rb = B.extraction.row( ib ).transpose();
                if( ( ra - rb ).norm() < coeff_tol ) o.identity_col_match++;
                if( ( ra - rb.reverse() ).norm() < coeff_tol ) o.reversed_col_match++;
            }
            out.push_back( o );
        }
        return out;
    }

    size_t dartOf( const MultiPatchCombinatorialMap& mp, const size_t patch, const size_t t )
    {
        return patchTopCellAt( mp, patch, { 0, t } ).dart().id();
    }

    std::vector<size_t> interfaceIds( const SplineSpace& ss,
                                      const MultiPatchCombinatorialMap& mp,
                                      const size_t patch,
                                      const size_t t )
    {
        const ElementSide side = ( patch == 0 ) ? ElementSide( 0, false ) : ElementSide( 0, true );
        const TraceSideData d = traceSideData( ss, patchTopCellAt( mp, patch, { 0, t } ), side );
        std::vector<size_t> ids;
        for( const FunctionId f : d.connectivity ) ids.push_back( f.id() );
        return ids;
    }

    MultiPatchSplineSpace makeSurface( const TPPermutation p )
    {
        const double ptol = 1e-10;
        const KnotVector kv_s( { 0, 0, 0, 1, 1, 1 }, ptol );
        const KnotVector kv_t_c0( { 0, 0, 0, 1, 1, 2, 2, 2 }, ptol );
        const auto patch = makePatch( { kv_s, kv_t_c0 }, { 2, 2 } );
        return buildH1MultiPatchSplineSpace(
            { patch, patch }, twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ), p ) );
    }

    MultiPatchSplineSpace makeVolume( const TPPermutation p )
    {
        const double ptol = 1e-10;
        const KnotVector kv( { 0, 0, 0, 1, 1, 1 }, ptol );
        const auto patch = makePatch( { kv, kv, kv }, { 2, 2, 2 } );
        return buildH1MultiPatchSplineSpace(
            { patch, patch }, twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ), p ) );
    }
}

TEST_CASE( "EXPERIMENT A: requested versus effective leaves at an asymmetric interface", "[phase0]" )
{
    const double ptol = 1e-10;
    const KnotVector coarse_s( { 0, 0, 0, 1, 2, 2, 2 }, ptol );
    const KnotVector fine_s( { 0, 0, 0, 0.5, 1, 1.5, 2, 2, 2 }, ptol );
    const KnotVector t( { 0, 0, 0, 1, 1, 1 }, ptol );

    // Identical patches on both sides: the item-1 "same mesh" condition.
    const auto coarse_patch = makePatch( { coarse_s, t }, { 2, 2 } );
    const auto fine_patch = makePatch( { fine_s, t }, { 2, 2 } );
    const InternalConnectionsMap connections =
        twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ), TPPermutation::Flip1d );

    auto coarse_h1 = std::make_shared<const MultiPatchSplineSpace>(
        buildH1MultiPatchSplineSpace( { coarse_patch, coarse_patch }, connections ) );
    auto fine_h1 = std::make_shared<const MultiPatchSplineSpace>(
        buildH1MultiPatchSplineSpace( { fine_patch, fine_patch }, connections ) );

    const auto& cc = dynamic_cast<const MultiPatchCombinatorialMap&>(
        coarse_h1->basisComplex().parametricAtlas().cmap() );
    const auto& fc = dynamic_cast<const MultiPatchCombinatorialMap&>(
        fine_h1->basisComplex().parametricAtlas().cmap() );

    // patch0 refined at its max-s interface; patch1 left COARSE at its min-s interface.
    const std::vector<std::vector<Cell>> requested = {
        { patchTopCellAt( cc, 0, { 0, 0 } ),
          patchTopCellAt( cc, 1, { 0, 0 } ),
          patchTopCellAt( cc, 1, { 1, 0 } ) },
        { patchTopCellAt( fc, 0, { 2, 0 } ),
          patchTopCellAt( fc, 0, { 3, 0 } ) } };

    REQUIRE( requested.at( 0 ).size() == 3 );
    REQUIRE( requested.at( 1 ).size() == 2 );

    // PERMANENT: the scalar hierarchical space accepts an asymmetric leaf
    // selection.  An exception here fails the test rather than being printed.
    const HierarchicalMultiPatchSplineSpace ss =
        buildHierarchicalSplineSpace( { coarse_h1, fine_h1 }, requested );

    const auto& hcmap = dynamic_cast<const HierarchicalMultiPatchCombinatorialMap&>(
        ss.basisComplex().parametricAtlas().cmap() );
    const std::vector<std::vector<Cell>> effective = leafElements( hcmap );

    // PERMANENT: the constructor must not rewrite the caller's leaf selection.
    REQUIRE( effective.size() == requested.size() );
    for( size_t lvl = 0; lvl < requested.size(); lvl++ )
    {
        INFO( "level " << lvl );
        CHECK( effective.at( lvl ).size() == requested.at( lvl ).size() );

        std::set<size_t> req_d, eff_d;
        for( const Cell& c : requested.at( lvl ) ) req_d.insert( c.dart().id() );
        for( const Cell& c : effective.at( lvl ) ) eff_d.insert( c.dart().id() );
        CHECK( req_d == eff_d );
    }
    CHECK( ss.numFunctions() == 24 );

    // TEMPORARY (item 1): the vector-conforming builders currently REJECT this
    // configuration.  When item 1 lands these become successful builds and this
    // whole block must be replaced.
    REQUIRE_THROWS_AS( buildHDivHierarchicalMultiPatchSplineSpace( ss ), std::invalid_argument );
    REQUIRE_THROWS_WITH( buildHDivHierarchicalMultiPatchSplineSpace( ss ),
                         Catch::Matchers::ContainsSubstring( "matching leaf elements" ) );
    REQUIRE_THROWS_AS( buildHCurlHierarchicalMultiPatchSplineSpace( ss ), std::invalid_argument );
    REQUIRE_THROWS_WITH( buildHCurlHierarchicalMultiPatchSplineSpace( ss ),
                         Catch::Matchers::ContainsSubstring( "matching leaf elements" ) );
}

TEST_CASE( "EXPERIMENT B: elementwise cross-patch correspondence, 2d Flip1d", "[phase0]" )
{
    const MultiPatchSplineSpace h1 = makeSurface( TPPermutation::Flip1d );
    const std::vector<PairObs> h1_obs = observeInterfaces( h1 );

    // PERMANENT: two elements along the interface give two paired interface
    // elements, each a 3-function quadratic trace on both sides.
    REQUIRE( h1_obs.size() == 2 );
    for( const PairObs& o : h1_obs )
    {
        INFO( "dartA=" << o.dart_a << " dartB=" << o.dart_b );
        CHECK( o.a_rows == 3 );
        CHECK( o.a_cols == 3 );
        CHECK( o.b_rows == 3 );
        CHECK( o.b_cols == 3 );
    }

    // PERMANENT: the identity correspondence shares all 3 trace functions per
    // paired element, with coefficients agreeing column for column.
    // reversed_col_match == 1 counts only the self-mirrored middle function and
    // is a fixture artifact, not an invariant of the convention.
    for( const PairObs& o : h1_obs )
    {
        INFO( "dartA=" << o.dart_a << " dartB=" << o.dart_b );
        CHECK( o.shared == 3 );
        CHECK( o.identity_col_match == 3 );
        CHECK( o.reversed_col_match == 1 );
    }

    // TEMPORARY (row D): with correct pairing the vector spaces share 2 of 5
    // per paired element rather than 0.  2 is NOT yet the right answer -- the
    // lower-dimensional H(curl)/H(div) merge is a separate open blocker -- so
    // these pin current behaviour and retire with row D.
    for( const PairObs& o : observeInterfaces( buildHDivMultiPatchSplineSpace( h1 ) ) )
    {
        CHECK( o.a_rows == 5 );
        CHECK( o.a_cols == 5 );
        CHECK( o.shared == 2 );
    }
    for( const PairObs& o : observeInterfaces( buildHCurlMultiPatchSplineSpace( h1 ) ) )
    {
        CHECK( o.a_rows == 5 );
        CHECK( o.a_cols == 5 );
        CHECK( o.shared == 2 );
    }
}

TEST_CASE( "EXPERIMENT B: elementwise cross-patch correspondence, 3d permutations", "[phase0]" )
{
    // NONDISCRIMINATING for pairing: one interface element per patch.  Column
    // order only.  h1/hdiv/hcurl identity and reversed column-match counts.
    struct Expect
    {
        TPPermutation p;
        size_t h1_id, h1_rev, hdiv_id, hdiv_rev, hcurl_id, hcurl_rev;
    };
    const std::vector<Expect> expects = {
        { TPPermutation::ZeroToZero, 9, 1, 4, 0, 12, 1 },
        { TPPermutation::ZeroToOne, 1, 1, 0, 0, 0, 0 },
        { TPPermutation::ZeroToTwo, 1, 9, 0, 0, 0, 0 },
        { TPPermutation::ZeroToThree, 1, 1, 0, 0, 0, 0 } };

    for( const Expect& e : expects )
    {
        INFO( "permutation " << permName( e.p ) );
        const MultiPatchSplineSpace h1 = makeVolume( e.p );

        const std::vector<PairObs> h1_obs = observeInterfaces( h1 );
        const std::vector<PairObs> hdiv_obs = observeInterfaces( buildHDivMultiPatchSplineSpace( h1 ) );
        const std::vector<PairObs> hcurl_obs = observeInterfaces( buildHCurlMultiPatchSplineSpace( h1 ) );

        // PERMANENT: a single interface element pair, fully shared on each space.
        REQUIRE( h1_obs.size() == 1 );
        REQUIRE( hdiv_obs.size() == 1 );
        REQUIRE( hcurl_obs.size() == 1 );
        CHECK( h1_obs.front().a_cols == 9 );
        CHECK( h1_obs.front().shared == 9 );
        CHECK( hdiv_obs.front().a_cols == 16 );
        CHECK( hdiv_obs.front().shared == 4 );
        CHECK( hcurl_obs.front().a_cols == 21 );
        CHECK( hcurl_obs.front().shared == 12 );

        // TEMPORARY (step e, reassigned from row D 2026-10-05): extraction
        // column order.  Only ZeroToZero agrees
        // under identity and only ZeroToTwo under reversal for H1; the vector
        // spaces agree under neither, so sign/direction handling is missing
        // beyond reordering.  MEASURED UNCHANGED by the Stage 2 transform:
        // column order comes from the extraction operator, not from element
        // pairing, and a 1x1 side pairs trivially under either convention.
        CHECK( h1_obs.front().identity_col_match == e.h1_id );
        CHECK( h1_obs.front().reversed_col_match == e.h1_rev );
        CHECK( hdiv_obs.front().identity_col_match == e.hdiv_id );
        CHECK( hdiv_obs.front().reversed_col_match == e.hdiv_rev );
        CHECK( hcurl_obs.front().identity_col_match == e.hcurl_id );
        CHECK( hcurl_obs.front().reversed_col_match == e.hcurl_rev );
    }
}

TEST_CASE( "EXPERIMENT B2: 2d Flip1d pairing versus merging diagnostic", "[phase0]" )
{
    const MultiPatchSplineSpace h1 = makeSurface( TPPermutation::Flip1d );
    const auto& mp = dynamic_cast<const MultiPatchCombinatorialMap&>(
        h1.basisComplex().parametricAtlas().cmap() );

    // PERMANENT: sides 0<->1 correspond by the identity, so element t pairs
    // with element t; F_a cancels Flip1d's reversal.  This row previously
    // pinned the side-blind consumer's reversed pairing and was mislabelled.
    const std::vector<PairObs> obs = observeInterfaces( h1 );
    REQUIRE( obs.size() == 2 );
    for( const PairObs& o : obs )
    {
        INFO( "dartA=" << o.dart_a << " dartB=" << o.dart_b );
        bool matched = false;
        for( size_t t = 0; t < 2; t++ )
        {
            if( o.dart_a == dartOf( mp, 0, t ) )
            {
                CHECK( o.dart_b == dartOf( mp, 1, t ) );
                matched = true;
            }
        }
        CHECK( matched );
    }

    // PERMANENT: patch1 element {0,t} carries exactly the global ids of patch0
    // element {0,t}, which AGREES with the identity pairing above.  These rows
    // address elements through pure topology and read merged ids, so the
    // consumer fix left them untouched; only their TEMPORARY label was wrong.
    for( size_t t = 0; t < 2; t++ )
    {
        INFO( "t=" << t );
        CHECK( interfaceIds( h1, mp, 1, t ) == interfaceIds( h1, mp, 0, t ) );
    }
    CHECK( h1.numFunctions() == 25 );
}

TEST_CASE( "EXPERIMENT B3: a self-glued asymmetric patch is identity-compatible on sides 0<->1", "[phase0]" )
{
    const double ptol = 1e-10;
    const KnotVector kv_s( { 0, 0, 0, 1, 1, 1 }, ptol );
    // Not a palindrome: reversing {0,0,0,1,3,3,3} gives {0,0,0,2,3,3,3}.
    const KnotVector kv_t_asym( { 0, 0, 0, 1, 3, 3, 3 }, ptol );
    const auto patch = makePatch( { kv_s, kv_t_asym }, { 2, 2 } );

    // PERMANENT: sides 0<->1 correspond by the identity, so a patch glued to
    // itself across a non-palindromic interface matches axis for axis and is
    // accepted.  The side-blind validator compared the pattern against its own
    // reversal and wrongly rejected this; B4 is the converse case.
    const MultiPatchSplineSpace h1 = buildH1MultiPatchSplineSpace(
        { patch, patch },
        twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ), TPPermutation::Flip1d ) );

    const auto& mp = dynamic_cast<const MultiPatchCombinatorialMap&>(
        h1.basisComplex().parametricAtlas().cmap() );

    // PERMANENT: identity pairing on the accepted interface.
    const std::vector<PairObs> obs = observeInterfaces( h1 );
    REQUIRE( obs.size() == 2 );
    for( const PairObs& o : obs )
    {
        INFO( "dartA=" << o.dart_a << " dartB=" << o.dart_b );
        bool matched = false;
        for( size_t t = 0; t < 2; t++ )
        {
            if( o.dart_a == dartOf( mp, 0, t ) )
            {
                CHECK( o.dart_b == dartOf( mp, 1, t ) );
                matched = true;
            }
        }
        CHECK( matched );
    }
}

TEST_CASE( "EXPERIMENT B4: mirrored asymmetric interfaces are identity-incompatible and rejected", "[phase0]" )
{
    const double ptol = 1e-10;
    const KnotVector kv_s( { 0, 0, 0, 1, 1, 1 }, ptol );
    // patch0 interface element sizes [1,2]; patch1 is its mirror, sizes [2,1].
    const KnotVector kv_t_A( { 0, 0, 0, 1, 3, 3, 3 }, ptol );
    const KnotVector kv_t_B( { 0, 0, 0, 2, 3, 3, 3 }, ptol );
    const auto patch_A = makePatch( { kv_s, kv_t_A }, { 2, 2 } );
    const auto patch_B = makePatch( { kv_s, kv_t_B }, { 2, 2 } );

    // PERMANENT: the fixture is the discriminating one.  Reversed pairing
    // couples spans of equal length; identity pairing couples unequal spans.
    const Eigen::VectorXd len_a = parametricLengths( kv_t_A );
    const Eigen::VectorXd len_b = parametricLengths( kv_t_B );
    REQUIRE( len_a.size() == 2 );
    REQUIRE( len_b.size() == 2 );
    for( Eigen::Index t = 0; t < len_a.size(); t++ )
    {
        INFO( "t=" << t );
        CHECK( std::abs( len_a( t ) - len_b( len_b.size() - 1 - t ) ) < geom_tol );
    }
    CHECK( std::abs( len_a( 0 ) - len_b( 0 ) ) > geom_tol );

    // PERMANENT: sides 0<->1 correspond by the identity, which would couple
    // patch0's size-1 span to patch1's size-2 span, so this interface is
    // genuinely incompatible and must be rejected.  Exact converse of B3:
    // mirroring reconciles the patterns only under a REVERSING side pair.
    REQUIRE_THROWS_AS(
        buildH1MultiPatchSplineSpace(
            { patch_A, patch_B },
            twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ), TPPermutation::Flip1d ) ),
        std::invalid_argument );
    REQUIRE_THROWS_WITH(
        buildH1MultiPatchSplineSpace(
            { patch_A, patch_B },
            twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ), TPPermutation::Flip1d ) ),
        Catch::Matchers::ContainsSubstring( "matching tangential spline parameterizations" ) );

    // PERMANENT: per-patch counts are convention-invariant and survive the
    // rejection.  The glued count h1.numFunctions() == 20 cannot live here now
    // that construction throws; it belongs to a fixture that still builds.
    CHECK( patch_A->numFunctions() == 12 );
    CHECK( patch_B->numFunctions() == 12 );
}

TEST_CASE( "EXPERIMENT B5: discriminating 3d interface, four elements per side", "[phase0]" )
{
    const double ptol = 1e-10;
    const KnotVector kv_n( { 0, 0, 0, 1, 1, 1 }, ptol );
    // Two elements on each tangential axis, neither palindromic, and the two
    // axes mutually distinct, so a reversal or an axis swap changes the pairing.
    const KnotVector kv_u( { 0, 0, 0, 1, 3, 3, 3 }, ptol ); // spans [1,2]
    const KnotVector kv_v( { 0, 0, 0, 2, 3, 3, 3 }, ptol ); // spans [2,1]
    const auto patch = makePatch( { kv_n, kv_u, kv_v }, { 2, 2, 2 } );

    // PERMANENT: on sides 0<->1 with ZeroToZero, F_a's reversal of tangential
    // axis 0 is cancelled by P's own flip, so the total map F_b o P o F_a is the
    // identity and a patch glued to itself matches forward on both tangential
    // axes.  The side-blind consumer applied P alone, which reverses tangential
    // axis 0, and rejected this fixture outright.
    const MultiPatchSplineSpace h1 = buildH1MultiPatchSplineSpace(
        { patch, patch },
        twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ),
                            TPPermutation::ZeroToZero ) );

    const auto& mp = dynamic_cast<const MultiPatchCombinatorialMap&>(
        h1.basisComplex().parametricAtlas().cmap() );

    const std::vector<PairObs> obs = observeInterfaces( h1 );
    REQUIRE( obs.size() == 4 );

    // PERMANENT: element (i,j) pairs with element (i,j), and the shared set is
    // the whole trace connectivity of that element, not merely its size.
    for( const PairObs& o : obs )
    {
        INFO( "dartA=" << o.dart_a << " dartB=" << o.dart_b );
        bool matched = false;
        for( size_t i = 0; i < 2; i++ )
        {
            for( size_t j = 0; j < 2; j++ )
            {
                if( o.dart_a == patchTopCellAt( mp, 0, { 0, i, j } ).dart().id() )
                {
                    CHECK( o.dart_b == patchTopCellAt( mp, 1, { 0, i, j } ).dart().id() );
                    matched = true;
                }
            }
        }
        CHECK( matched );
        CHECK( o.a_cols == 9 );
        CHECK( o.shared == 9 );
    }
}

TEST_CASE( "EXPERIMENT B6: mirrored 3d tangential knots are rejected on this side pair",
           "[phase0]" )
{
    const double ptol = 1e-10;
    const KnotVector kv_n( { 0, 0, 0, 1, 1, 1 }, ptol );
    const KnotVector kv_u( { 0, 0, 0, 1, 3, 3, 3 }, ptol );
    const KnotVector kv_v( { 0, 0, 0, 2, 3, 3, 3 }, ptol );
    const auto patch_A = makePatch( { kv_n, kv_u, kv_v }, { 2, 2, 2 } );
    const auto patch_B = makePatch( { kv_n, kv_v, kv_v }, { 2, 2, 2 } );

    // PERMANENT: the identity total map couples tangential axis 1 forward, so
    // mirrored patterns [1,2] against [2,1] are genuinely incompatible.
    // Converse of B5 and the 3d counterpart of B4.
    REQUIRE_THROWS_AS(
        buildH1MultiPatchSplineSpace(
            { patch_A, patch_B },
            twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ),
                                TPPermutation::ZeroToZero ) ),
        std::invalid_argument );
}
