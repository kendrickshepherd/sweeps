// Row D diagnostics.  See outstanding_mathematics.txt items M4, M5, M9, M10.
//
// EXPERIMENT 1 (M10): is the H(div)/H(curl) per-paired-element shared count of
// 2 the correct degree-dependent trace count, or a defect symptom?  Measured
// across degrees on BOTH a reversing and a non-reversing side pair -- a count
// measured on one orientation class only would repeat the Stage 2 defect.
#include <catch2/catch_test_macros.hpp>
#include <KnotVector.hpp>
#include <MultiPatchSplineFactory.hpp>
#include <TraceMesh.hpp>
#include <CombinatorialMapMethods.hpp>
#include <array>
#include <map>
#include <optional>
#include <string>
#include <VectorConformingMultiPatchSplineSpace.hpp>
#include <algorithm>
#include <limits>
#include <iostream>
#include <set>
#include <vector>
#include <MultiPatchParametricAtlas.hpp>
#include <ParametricAtlas1d.hpp>
#include <CombinatorialMap1d.hpp>
#include <ParametricAtlas.hpp>
#include <SideCoordinateTransform.hpp>
#include <CommonUtils.hpp>
#include <IndexOperations.hpp>
#include <UnionFind.hpp>
#include <VectorConformingTPSplineSpace.hpp>
#include <tuple>
#include <sstream>
#include <limits>

using namespace basis;
using namespace topology;

namespace
{
    using InternalConnectionsMap = MultiPatchCombinatorialMap::InternalConnectionsMap;
    using ConstituentSide = MultiPatchCombinatorialMap::ConstituentSide;
    using TPPermutation = MultiPatchCombinatorialMap::TPPermutation;

    constexpr double ptol = 1e-10;
    constexpr double coeff_tol = 1e-10;

    // Open knot vector, interior multiplicity = degree (C0 joints), so n_elems
    // spans regardless of degree.
    KnotVector openKV( const size_t degree, const size_t n_elems )
    {
        std::vector<double> knots( degree + 1, 0.0 );
        for( size_t e = 1; e < n_elems; e++ )
            knots.insert( knots.end(), degree, static_cast<double>( e ) );
        knots.insert( knots.end(), degree + 1, static_cast<double>( n_elems ) );
        return KnotVector( knots, ptol );
    }

    std::shared_ptr<const TPSplineSpace> makePatch( const SmallVector<KnotVector, 3>& kvs,
                                                    const SmallVector<size_t, 3>& degrees )
    {
        return std::make_shared<const TPSplineSpace>( buildBSpline( kvs, degrees ) );
    }

    InternalConnectionsMap twoPatchConnection( const ElementSide& first_side,
                                               const ElementSide& second_side,
                                               const TPPermutation permutation )
    {
        const ConstituentSide first{ 0, first_side.sideId() };
        const ConstituentSide second{ 1, second_side.sideId() };
        return { { first, { permutation, second } }, { second, { permutation, first } } };
    }

    // Non-reversing class: sides 0<->1.  [a in {1,2}] + [b in {1,2}] is ODD, so
    // F_a cancels the hardcoded Flip1d and the correspondence is the identity.
    // The interface is patch-axis 1 on both sides, carrying the 2-element knots.
    MultiPatchSplineSpace makeNonReversing( const size_t degree )
    {
        const auto patch = makePatch( { openKV( degree, 1 ), openKV( degree, 2 ) }, { degree, degree } );
        return buildH1MultiPatchSplineSpace(
            { patch, patch },
            twoPatchConnection( ElementSide( 0, false ), ElementSide( 0, true ), TPPermutation::Flip1d ) );
    }

    // Reversing class: sides 0<->3.  Parity is EVEN, so the correspondence is
    // reversed.  Patch 0's side 0 is tangential in axis 1; patch 1's side 3 is
    // tangential in axis 0, so patch 1 swaps its axes to keep the tangential
    // knot vectors equal.
    MultiPatchSplineSpace makeReversing( const size_t degree )
    {
        const auto a = makePatch( { openKV( degree, 1 ), openKV( degree, 2 ) }, { degree, degree } );
        const auto b = makePatch( { openKV( degree, 2 ), openKV( degree, 1 ) }, { degree, degree } );
        return buildH1MultiPatchSplineSpace(
            { a, b },
            twoPatchConnection( ElementSide( 0, false ), ElementSide( 1, true ), TPPermutation::Flip1d ) );
    }

    struct PairObs
    {
        Eigen::Index a_rows = 0;
        Eigen::Index a_cols = 0;
        Eigen::Index b_rows = 0;
        Eigen::Index b_cols = 0;
        size_t shared = 0;
        size_t identity_col_match = 0;
        size_t reversed_col_match = 0;
        // Negated variants: an H(curl) tangential dof must flip sign across a
        // reversing interface, so agreement there is up to a minus.
        size_t identity_col_match_neg = 0;
        size_t reversed_col_match_neg = 0;
    };

    std::vector<PairObs> observeInterfaces( const SplineSpace& ss )
    {
        std::vector<PairObs> out;
        for( const TraceMeshInterface& iface : patchTraceMeshInterfaces( ss ) )
        {
            const TraceInterfaceElement tr = traceInterfaceElement( ss, iface );
            REQUIRE( tr.second.has_value() );
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
                if( ( ra + rb ).norm() < coeff_tol ) o.identity_col_match_neg++;
                if( ( ra + rb.reverse() ).norm() < coeff_tol ) o.reversed_col_match_neg++;
            }
            out.push_back( o );
        }
        return out;
    }

    void report( const char* label, const size_t degree, const char* space,
                 const std::vector<PairObs>& obs )
    {
        std::cout << "  " << label << "  p=" << degree << "  " << space
                  << "  pairs=" << obs.size();
        for( const PairObs& o : obs )
            std::cout << "  [rows=" << o.a_rows << "/" << o.b_rows
                      << " cols=" << o.a_cols << "/" << o.b_cols
                      << " shared=" << o.shared
                      << " idcol=" << o.identity_col_match
                      << " revcol=" << o.reversed_col_match
                      << " -idcol=" << o.identity_col_match_neg
                      << " -revcol=" << o.reversed_col_match_neg << "]";
        std::cout << std::endl;
    }

    // The ordering and sign the shared extraction rows are expected to agree
    // under.  ASSERTED: that channel matches every shared row, and its
    // opposite-sign counterpart matches none.  Deliberately NOT asserted: that
    // the other two comparisons vanish.  A symmetric scalar basis function is
    // its own mirror, so H1 shows one incidental match under the other
    // ordering whenever p is even.
    enum class Match
    {
        Identity,
        Reversed,
        NegatedReversed
    };

    void checkPattern( const std::vector<PairObs>& obs,
                       const Eigen::Index expected_trace_fns,
                       const size_t expected_shared,
                       const Match match )
    {
        REQUIRE( obs.size() == 2 ); // PERMANENT: two interface elements.
        for( const PairObs& o : obs )
        {
            CHECK( o.a_rows == expected_trace_fns );
            CHECK( o.b_rows == expected_trace_fns );
            CHECK( o.a_cols == expected_trace_fns );
            CHECK( o.b_cols == expected_trace_fns );
            CHECK( o.shared == expected_shared );
            switch( match )
            {
                case Match::Identity:
                    CHECK( o.identity_col_match == expected_shared );
                    CHECK( o.identity_col_match_neg == 0 );
                    break;
                case Match::Reversed:
                    CHECK( o.reversed_col_match == expected_shared );
                    CHECK( o.reversed_col_match_neg == 0 );
                    break;
                case Match::NegatedReversed:
                    CHECK( o.reversed_col_match_neg == expected_shared );
                    CHECK( o.reversed_col_match == 0 );
                    break;
            }
        }
    }

    void sweepOne( const char* label,
                   const MultiPatchSplineSpace& h1,
                   const size_t degree,
                   const Match scalar_match,
                   const Match hdiv_match,
                   const Match hcurl_match )
    {
        const std::vector<PairObs> h1_obs = observeInterfaces( h1 );
        const std::vector<PairObs> hdiv_obs = observeInterfaces( buildHDivMultiPatchSplineSpace( h1 ) );
        const std::vector<PairObs> hcurl_obs = observeInterfaces( buildHCurlMultiPatchSplineSpace( h1 ) );
        report( label, degree, "H1   ", h1_obs );
        report( label, degree, "HDiv ", hdiv_obs );
        report( label, degree, "HCurl", hcurl_obs );

        const auto p = static_cast<Eigen::Index>( degree );
        // PERMANENT: a dimension count, not a fit.  The scalar trace carries
        // degree p, so p+1 functions, all continuous.  Each vector trace
        // carries 2p+1 = p + (p+1): the conforming component has degree p-1
        // and contributes p shared functions, the other has degree p and
        // shares none.
        INFO( label << " p=" << degree );
        checkPattern( h1_obs, p + 1, degree + 1, scalar_match );
        checkPattern( hdiv_obs, 2 * p + 1, degree, hdiv_match );
        checkPattern( hcurl_obs, 2 * p + 1, degree, hcurl_match );
    }
}

// MEASURED 2026-10-05, confirming the conjecture recorded at 8d1e4ce before the
// sweep ran.  shared == p exactly, identically on both orientation classes, so
// the count is degree-correct and not a defect symptom.  This retires
// shared == 2 as row D's marker; it does NOT close row D, which is a
// codimension-two 3d question no 2d measurement can reach.
//
// The sign pattern below was unplanned.  H(curl) across a reversing interface
// agrees under reversal AND negation, with nothing matching in the non-negated
// reversed channel -- consistent with the expected de Rham orientation, since
// the two patches induce antiparallel tangents there and H(curl)'s conforming
// component is the tangential one.  H(div)'s is the normal and needs no flip.
TEST_CASE( "EXPERIMENT 1: degree dependence of the 2d interface shared count", "[rowd]" )
{
    std::cout << "EXPERIMENT 1: 2d per-paired-element shared trace counts" << std::endl;
    for( const size_t degree : { 2ul, 3ul, 4ul } )
    {
        sweepOne( "non-reversing 0<->1", makeNonReversing( degree ), degree,
                  Match::Identity, Match::Identity, Match::Identity );
        sweepOne( "reversing     0<->3", makeReversing( degree ), degree,
                  Match::Reversed, Match::Reversed, Match::NegatedReversed );
    }
}

// ---------------------------------------------------------------------------
// STEP 5: topology assertions, BEFORE any vector space is constructed.
// Covers M4's required assertions 1, 2 and 5.  Assertion 6 needs the
// function-id merge and stays open; it is a step 6 item.  Assertion 5 is
// load-bearing:
// by M4's tree corollary, on a graph with beta_1 == 0 every 1-cochain is a
// coboundary, so a fixture whose edge has a tree-shaped incident-patch graph
// CANNOT fail the orientation parity test and proves nothing about it.
// ---------------------------------------------------------------------------
namespace
{
    using DartConnections = std::map<std::pair<size_t, Dart>, std::pair<size_t, Dart>>;

    std::shared_ptr<const MultiPatchCombinatorialMap>
        buildMultiPatch3d( const size_t n_patches, const DartConnections& conns )
    {
        const KnotVector kv( { 0, 0, 0, 1, 1, 1 }, ptol );
        const auto ss_tp = std::make_shared<const TPSplineSpace>(
            buildBSpline( { kv, kv, kv }, { 2, 2, 2 } ) );
        const auto cmap_tp = ss_tp->basisComplexPtr()->parametricAtlasPtr()->cmapPtr();
        return std::make_shared<const MultiPatchCombinatorialMap>(
            std::vector<std::shared_ptr<const TPCombinatorialMap>>( n_patches, cmap_tp ), conns );
    }

    // The fixture from MultiPatchSplineSpace_test.cpp:325, topology only.
    DartConnections fourPatchConnections()
    {
        return { { { 0, Dart( 0 ) }, { 1, Dart( 5 ) } },
                 { { 0, Dart( 8 ) }, { 2, Dart( 5 ) } },
                 { { 0, Dart( 16 ) }, { 3, Dart( 23 ) } },
                 { { 1, Dart( 8 ) }, { 2, Dart( 22 ) } },
                 { { 1, Dart( 16 ) }, { 3, Dart( 2 ) } },
                 { { 2, Dart( 16 ) }, { 3, Dart( 8 ) } } };
    }

    // The same four patches with the 0-1 interface re-glued a quarter turn, so
    // that its side coordinate transform reverses a tangential axis.  Chosen by
    // exhaustive dart search: of the 1536 single-slot perturbations that keep
    // all six interfaces, 1152 reverse some axis and 40 of those both place a
    // reversing interface on a non-tree edge and admit the spline spaces.  M4
    // assertion 4 is only discriminating on a reversing fixture like this one.
    // Assertion 6 needs more and is NOT met here; see reversingRingConnections.
    DartConnections reversingFourPatchConnections()
    {
        DartConnections conns = fourPatchConnections();
        conns.erase( std::pair<size_t, Dart>{ 0, Dart( 0 ) } );
        conns.emplace( std::pair<size_t, Dart>{ 0, Dart( 1 ) }, std::pair<size_t, Dart>{ 1, Dart( 3 ) } );
        return conns;
    }

    // A three-patch RING around one common edge, with two of its three
    // interfaces reversing the edge direction.  A MINIMAL fixture satisfying
    // M4 assertion 6, not the only cyclic topology that could; what it has to
    // supply is structural.  An H(curl) component is conforming on a face only
    // when its index is a tangential axis of that face, and a patch meets an
    // edge in two faces with distinct normals, so the ONLY component
    // constrained across both is the one along the edge direction.  A cycle of
    // G at an edge junction is therefore built from edge-direction DOFs alone,
    // and a flip on that cycle needs the reversal to fall on the
    // edge-direction axis.  Reversing some other tangential axis - as the
    // four-patch fixture above does - produces flips that are all bridges.
    // Found by exhaustive dart search over rings.
    DartConnections reversingRingConnections()
    {
        return { { { 0, Dart( 7 ) }, { 1, Dart( 19 ) } },
                 { { 1, Dart( 0 ) }, { 2, Dart( 3 ) } },
                 { { 2, Dart( 7 ) }, { 0, Dart( 12 ) } } };
    }

    using PatchPair = std::pair<size_t, size_t>;

    // The two ConstituentSides of one declared interface, as plain indices
    // {p, s, q, t} canonicalised so (p,s) <= (q,t).  Kept because a patch pair
    // alone cannot name an interface: two distinct faces joining the same two
    // patches would collapse into a single graph arc.
    using SideKey = std::array<size_t, 4>;

    SideKey makeSideKey( const ConstituentSide& a, const ConstituentSide& b )
    {
        const SideKey x{ a.constituent_id, a.side_id, b.constituent_id, b.side_id };
        const SideKey y{ b.constituent_id, b.side_id, a.constituent_id, a.side_id };
        return std::min( x, y );
    }

    // PRECONDITION for reading an incident-patch graph off patch pairs: at most
    // one declared interface may join any ordered patch pair.  Asserted per
    // fixture rather than assumed.  Step 6 carries side identity explicitly and
    // must not rely on it.
    size_t maxInterfacesPerPatchPair( const MultiPatchCombinatorialMap& mp )
    {
        std::map<PatchPair, size_t> counts;
        for( const auto& [from, to] : mp.connections() )
            counts[{ from.constituent_id, to.second.constituent_id }]++;
        size_t worst = 0;
        for( const auto& [patch_pair, n] : counts ) worst = std::max( worst, n );
        return worst;
    }

    // The declared interface joining two patches, or nullopt if there is not
    // exactly one.
    std::optional<SideKey>
        interfaceBetween( const MultiPatchCombinatorialMap& mp, const size_t p, const size_t q )
    {
        std::optional<SideKey> found;
        for( const auto& [from, to] : mp.connections() )
        {
            if( from.constituent_id != p or to.second.constituent_id != q ) continue;
            if( found.has_value() ) return std::nullopt;
            found = makeSideKey( from, to.second );
        }
        return found;
    }

    std::set<PatchPair> patchPairsOf( const InternalConnectionsMap& conns )
    {
        std::set<PatchPair> out;
        for( const auto& [from, to] : conns )
            out.insert( { std::min( from.constituent_id, to.second.constituent_id ),
                          std::max( from.constituent_id, to.second.constituent_id ) } );
        return out;
    }

    struct EdgeObs
    {
        size_t n_darts = 0;
        std::set<size_t> patches;        // incident patches (graph vertices)
        std::set<PatchPair> transitions; // patch-level arcs this edge traverses
        std::set<SideKey> side_transitions; // the same arcs, with face identity
        bool connected = false;
        size_t beta_1 = 0;
    };

    // The incident-patch graph of one edge, read off the REALIZED map rather
    // than off the declared connection list: a vertex per incident patch, an
    // arc per cross-patch phi_3 transition the edge's own orbit traverses.
    EdgeObs observeEdge( const MultiPatchCombinatorialMap& mp, const Cell& edge )
    {
        EdgeObs o;
        iterateDartsOfCell( mp, edge, [&]( const Dart& d ) {
            o.n_darts++;
            const auto [p, local_d] = mp.toLocalDart( d );
            o.patches.insert( p );
            const std::optional<Dart> nb = phi( mp, 3, d );
            if( nb.has_value() )
            {
                const auto [q, local_q] = mp.toLocalDart( nb.value() );
                if( q != p ) o.transitions.insert( { std::min( p, q ), std::max( p, q ) } );
            }
            return true;
        } );

        // Resolve each patch-level arc to the declared interface it crosses.
        // side_transitions.size() < transitions.size() means some patch pair
        // carries more than one interface and the patch-level graph is lossy.
        for( const PatchPair& t : o.transitions )
        {
            const std::optional<SideKey> k = interfaceBetween( mp, t.first, t.second );
            if( k.has_value() ) o.side_transitions.insert( k.value() );
        }

        // Connectivity by BFS over the transition arcs.
        if( not o.patches.empty() )
        {
            std::set<size_t> seen{ *o.patches.begin() };
            std::vector<size_t> stack{ *o.patches.begin() };
            while( not stack.empty() )
            {
                const size_t v = stack.back();
                stack.pop_back();
                for( const PatchPair& e : o.transitions )
                {
                    const size_t other = ( e.first == v ) ? e.second : ( e.second == v ? e.first : v );
                    if( other != v and seen.insert( other ).second ) stack.push_back( other );
                }
            }
            o.connected = ( seen.size() == o.patches.size() );
        }
        // beta_1 = E - V + C, with C == 1 when connected.
        if( o.connected and o.transitions.size() + 1 >= o.patches.size() )
            o.beta_1 = o.transitions.size() + 1 - o.patches.size();
        return o;
    }

    // Internal topology-orbit accounting -- NOT an independent merge oracle,
    // since cellCount and observeEdge both run on the same orbit machinery.
    // What it does check: the observed global edge orbits account for every
    // patch-local edge, with no unexpected same-patch multiplicity.
    std::pair<size_t, size_t> edgeAccounting( const MultiPatchCombinatorialMap& mp )
    {
        const size_t n_global = cellCount( mp, 1 );
        size_t merges = 0;
        iterateCellsWhile( mp, 1, [&]( const Cell& edge ) {
            merges += observeEdge( mp, edge ).patches.size() - 1;
            return true;
        } );
        return { n_global, merges };
    }
}

TEST_CASE( "STEP 5: four-patch 3d fixture topology", "[rowd]" )
{
    const DartConnections declared = fourPatchConnections();
    const auto mp = buildMultiPatch3d( 4, declared );

    std::cout << "STEP 5: four-patch 3d fixture" << std::endl;
    std::cout << "  declared face pairs = " << declared.size()
              << ", connections() entries = " << mp->connections().size() << std::endl;
    for( const auto& [from, to] : mp->connections() )
        std::cout << "    " << from << " -> " << to.second << "  " << to.first << std::endl;

    // M4 assertion 1 (M9): each declared pair must install BOTH directions.
    // initializeInterMapConnections uses emplace, which silently drops a
    // duplicate key, so a short count means a declaration was lost.
    CHECK( mp->connections().size() == 2 * declared.size() );

    // Precondition for the patch-level graph below.
    CHECK( maxInterfacesPerPatchPair( *mp ) == 1 );

    std::map<std::string, size_t> histogram;
    std::vector<EdgeObs> multi_patch_edges;
    iterateCellsWhile( *mp, 1, [&]( const Cell& edge ) {
        const EdgeObs o = observeEdge( *mp, edge );
        histogram[std::to_string( o.patches.size() ) + " patches, " +
                  std::to_string( o.transitions.size() ) + " arcs, beta_1=" +
                  std::to_string( o.beta_1 ) + ( o.connected ? ", connected" : ", DISCONNECTED" )]++;
        if( o.patches.size() > 1 ) multi_patch_edges.push_back( o );
        return true;
    } );

    std::cout << "  edges by incident-patch graph shape:" << std::endl;
    for( const auto& [shape, count] : histogram )
        std::cout << "    " << count << " x  " << shape << std::endl;

    std::vector<EdgeObs> cyclic;
    std::copy_if( multi_patch_edges.begin(), multi_patch_edges.end(), std::back_inserter( cyclic ),
                  []( const EdgeObs& o ) { return o.beta_1 > 0; } );
    std::cout << "  multi-patch edges = " << multi_patch_edges.size()
              << ", of which beta_1 > 0: " << cyclic.size() << std::endl;

    // MEASURED: the topology is K4 -- four single-element patches, each glued
    // to all three others on exactly three of its six faces.
    std::map<size_t, size_t> glued_sides_per_patch;
    for( const auto& [from, to] : mp->connections() ) glued_sides_per_patch[from.constituent_id]++;
    REQUIRE( glued_sides_per_patch.size() == 4 );
    for( const auto& [patch, n_glued] : glued_sides_per_patch )
    {
        INFO( "patch " << patch );
        CHECK( n_glued == 3 );
    }

    // M4 assertion 5, THE LOAD-BEARING ONE: the fixture contains edges whose
    // incident-patch graph is a cycle, so the orientation parity test is
    // actually exercised.  Each of the four patch triples of K4 shares exactly
    // one such edge: 3 patches, 3 arcs, connected, beta_1 == 1.
    CHECK( cyclic.size() == 4 );
    std::set<std::set<size_t>> cyclic_triples;
    std::set<PatchPair> arcs_on_cycles;
    for( const EdgeObs& o : cyclic )
    {
        CHECK( o.patches.size() == 3 );
        CHECK( o.transitions.size() == 3 );
        CHECK( o.connected );
        CHECK( o.beta_1 == 1 );
        // Face identity is preserved: one declared interface per arc.
        CHECK( o.side_transitions.size() == o.transitions.size() );
        cyclic_triples.insert( o.patches );
        arcs_on_cycles.insert( o.transitions.begin(), o.transitions.end() );
    }
    const std::set<std::set<size_t>> expected_triples{ { 0, 1, 2 }, { 0, 1, 3 }, { 0, 2, 3 }, { 1, 2, 3 } };
    CHECK( cyclic_triples == expected_triples );

    // M4 assertion 2: every declared face adjacency is contained in at least
    // one of the selected cyclic edges, so no declaration goes unexercised.
    // In K4 each pair lies on two of the four triples.
    std::set<PatchPair> declared_pairs;
    for( const auto& [left, right] : declared )
        declared_pairs.insert( { std::min( left.first, right.first ), std::max( left.first, right.first ) } );
    CHECK( declared_pairs.size() == 6 );
    CHECK( arcs_on_cycles == declared_pairs );

    // Topology-orbit accounting: 4 single-element hexes contribute 12 local
    // edges each, and the observed orbits must account for all of them.
    const auto [n_global_edges, merges] = edgeAccounting( *mp );
    std::cout << "  global edges = " << n_global_edges << ", identifications = " << merges << std::endl;
    CHECK( n_global_edges == 28 );
    CHECK( merges == 20 );
    CHECK( n_global_edges + merges == 4 * 12 );
}

namespace
{
    std::shared_ptr<const MultiPatchCombinatorialMap>
        buildMultiPatch3dFromSides( const size_t n_patches, const InternalConnectionsMap& conns )
    {
        const KnotVector kv( { 0, 0, 0, 1, 1, 1 }, ptol );
        const auto ss_tp = std::make_shared<const TPSplineSpace>(
            buildBSpline( { kv, kv, kv }, { 2, 2, 2 } ) );
        const auto cmap_tp = ss_tp->basisComplexPtr()->parametricAtlasPtr()->cmapPtr();
        return std::make_shared<const MultiPatchCombinatorialMap>(
            std::vector<std::shared_ptr<const TPCombinatorialMap>>( n_patches, cmap_tp ), conns );
    }

    // Three quads fanned around a shared corner: patch i's side 0 (s=0) glues
    // to patch i+1's side 2 (t=0), and both contain the local (0,0) corner, so
    // all three corners merge into one valence-three vertex and the ring
    // closes.  Swept, that vertex becomes a valence-three edge with beta_1 = 1.
    InternalConnectionsMap threePatchRing2d()
    {
        InternalConnectionsMap out;
        for( size_t i = 0; i < 3; i++ )
        {
            const ConstituentSide a{ i, 0 };
            const ConstituentSide b{ ( i + 1 ) % 3, 2 };
            out.emplace( a, std::pair<TPPermutation, ConstituentSide>{ TPPermutation::Flip1d, b } );
            out.emplace( b, std::pair<TPPermutation, ConstituentSide>{ TPPermutation::Flip1d, a } );
        }
        return out;
    }

    void reportEdgeShapes( const MultiPatchCombinatorialMap& mp, std::vector<EdgeObs>& cyclic_out )
    {
        std::map<std::string, size_t> histogram;
        iterateCellsWhile( mp, 1, [&]( const Cell& edge ) {
            const EdgeObs o = observeEdge( mp, edge );
            histogram[std::to_string( o.patches.size() ) + " patches, " +
                      std::to_string( o.transitions.size() ) + " arcs, beta_1=" +
                      std::to_string( o.beta_1 ) + ( o.connected ? ", connected" : ", DISCONNECTED" )]++;
            if( o.beta_1 > 0 ) cyclic_out.push_back( o );
            return true;
        } );
        std::cout << "  edges by incident-patch graph shape:" << std::endl;
        for( const auto& [shape, count] : histogram )
            std::cout << "    " << count << " x  " << shape << std::endl;
    }
}

TEST_CASE( "STEP 5: swept three-patch 3d fixture topology", "[rowd]" )
{
    const InternalConnectionsMap conns_2d = threePatchRing2d();
    const InternalConnectionsMap conns_3d = connectionsOfSweptMultipatch( conns_2d );
    const auto mp = buildMultiPatch3dFromSides( 3, conns_3d );

    std::cout << "STEP 5: swept three-patch ring" << std::endl;
    std::cout << "  2d connections = " << conns_2d.size()
              << ", swept 3d connections = " << conns_3d.size()
              << ", realized = " << mp->connections().size() << std::endl;
    for( const auto& [from, to] : mp->connections() )
        std::cout << "    " << from << " -> " << to.second << "  " << to.first << std::endl;

    // M4 assertion 1 for the swept producer: three declared 2d gluings, both
    // directions each, carried through the sweep without loss.
    CHECK( conns_2d.size() == 6 );
    CHECK( conns_3d.size() == 6 );
    CHECK( mp->connections().size() == 6 );

    std::vector<EdgeObs> cyclic;
    reportEdgeShapes( *mp, cyclic );
    std::cout << "  cyclic edges = " << cyclic.size() << std::endl;

    // Precondition for the patch-level graph.
    CHECK( maxInterfacesPerPatchPair( *mp ) == 1 );

    // M4 assertion 5 for this fixture: the swept central vertex is the one
    // valence-three edge, and its incident-patch graph is a 3-cycle.
    REQUIRE( cyclic.size() == 1 );
    const EdgeObs& central = cyclic.front();
    CHECK( central.patches.size() == 3 );
    CHECK( central.transitions.size() == 3 );
    CHECK( central.side_transitions.size() == 3 );
    CHECK( central.connected );
    CHECK( central.beta_1 == 1 );
    CHECK( central.patches == std::set<size_t>{ 0, 1, 2 } );

    // M4 assertion 2: every declared face pair occurs on that cyclic edge.
    const std::set<PatchPair> declared_pairs = patchPairsOf( conns_3d );
    CHECK( declared_pairs.size() == 3 );
    CHECK( central.transitions == declared_pairs );

    // Topology-orbit accounting: 3 single-element hexes, 12 local edges each.
    const auto [n_global_edges, merges] = edgeAccounting( *mp );
    std::cout << "  global edges = " << n_global_edges << ", identifications = " << merges << std::endl;
    CHECK( n_global_edges == 25 );
    CHECK( merges == 11 );
    CHECK( n_global_edges + merges == 3 * 12 );
}

// ---------------------------------------------------------------------------
// STEP 6: the H(curl) face-level dof constraints, their global ids and their
// orientations.  Covers M4's required assertions 3, 4 and 6, on both of the
// step 5 fixtures.
//
// The expected constraint set is derived from sideCoordinateTransform plus the
// de Rham tangential rule.  It reads neither coordinateTransform nor
// getIterVars -- the two production paths under examination (M5).  Every
// constraint is keyed by the actual pair of ConstituentSides, so nothing here
// inherits step 5's one-interface-per-patch-pair simplification.
// ---------------------------------------------------------------------------

namespace
{
    // tangentialAxes, restated from MultiPatchSplineSpace.cpp:18, which is
    // file-local there.  Side-local tangential axis d is the d-th patch axis
    // other than the side's normal axis, ascending.
    std::vector<size_t> tangentialAxes( const size_t dim, const size_t side_id )
    {
        std::vector<size_t> axes;
        for( size_t axis = 0; axis < dim; axis++ )
            if( axis != side_id / 2 ) axes.push_back( axis );
        return axes;
    }

    // Per-axis function counts of every vector component of one patch.
    using ComponentLengths = std::vector<std::vector<size_t>>;

    ComponentLengths componentLengths( const VectorConformingTPSplineSpace& ss )
    {
        ComponentLengths out;
        for( size_t k = 0; k < ss.numVectorComponents(); k++ )
        {
            std::vector<size_t> lengths;
            for( const auto& basis_1d : tensorProductComponentSplines( *ss.scalarTPBases().at( k ) ) )
                lengths.push_back( basis_1d->numFunctions() );
            out.push_back( lengths );
        }
        return out;
    }

    size_t blockSize( const std::vector<size_t>& lengths )
    {
        size_t n = 1;
        for( const size_t m : lengths ) n *= m;
        return n;
    }

    // Patch-local function id: the components are consecutive blocks, each
    // flattened over its own per-axis index.  Restated from the offset
    // arithmetic in VectorConformingMultiPatchSplineSpace.cpp.
    size_t localFid( const ComponentLengths& lengths, const size_t comp, const std::vector<size_t>& index )
    {
        size_t offset = 0;
        for( size_t k = 0; k < comp; k++ ) offset += blockSize( lengths.at( k ) );

        util::IndexVec iv, lens;
        for( size_t a = 0; a < index.size(); a++ )
        {
            iv.push_back( index.at( a ) );
            lens.push_back( lengths.at( comp ).at( a ) );
        }
        return offset + util::flatten( iv, lens );
    }

    // Side extents of one component on one side, in tangentialAxes order.
    std::vector<size_t> sideExtents( const ComponentLengths& lengths, const size_t comp, const size_t side_id )
    {
        std::vector<size_t> out;
        for( const size_t a : tangentialAxes( lengths.at( comp ).size(), side_id ) )
            out.push_back( lengths.at( comp ).at( a ) );
        return out;
    }

    // The face-supported functions of one component on one side, keyed by side
    // index.  The knot vectors are open with interior multiplicity p, so
    // exactly one function is nonzero at each end of an axis and the normal
    // index is pinned.
    std::map<std::vector<size_t>, size_t>
        faceFunctions( const ComponentLengths& lengths, const size_t comp, const size_t side_id )
    {
        const std::vector<size_t>& lens = lengths.at( comp );
        const size_t normal_axis = side_id / 2;
        // Side ids run S1, S0, T1, T0, U1, U0 -- the EVEN id of an axis is its
        // upper side.  See MultiPatchCombinatorialMap.cpp:25.
        const size_t pinned = ( side_id % 2 == 0 ) ? lens.at( normal_axis ) - 1 : 0;
        const std::vector<size_t> tang = tangentialAxes( lens.size(), side_id );

        std::map<std::vector<size_t>, size_t> out;
        for( size_t i = 0; i < lens.at( tang.at( 0 ) ); i++ )
        {
            for( size_t j = 0; j < lens.at( tang.at( 1 ) ); j++ )
            {
                std::vector<size_t> index( lens.size(), 0 );
                index.at( normal_axis ) = pinned;
                index.at( tang.at( 0 ) ) = i;
                index.at( tang.at( 1 ) ) = j;
                out.emplace( std::vector<size_t>{ i, j }, localFid( lengths, comp, index ) );
            }
        }
        return out;
    }

    // One intended face-level constraint, independently derived.
    struct Constraint
    {
        SideKey side_key{};
        size_t patch_a = 0, local_a = 0, comp_a = 0;
        size_t patch_b = 0, local_b = 0, comp_b = 0;
        bool aligned = true; // expected relative sign; true == same sign
    };

    // The face-level pairings between one component on side a and one component
    // on side b, under an already-derived side coordinate transform.
    std::vector<Constraint> pairFaceFunctions( const ConstituentSide& a,
                                               const SideCoordinateTransform& t,
                                               const ConstituentSide& b,
                                               const ComponentLengths& la,
                                               const ComponentLengths& lb,
                                               const size_t comp_a,
                                               const size_t comp_b,
                                               const bool aligned )
    {
        const std::vector<size_t> ext_a = sideExtents( la, comp_a, a.side_id );
        const std::vector<size_t> ext_b = sideExtents( lb, comp_b, b.side_id );
        // The reduced-degree direction has to line up under the transform.
        REQUIRE( transformedExtents( t, ext_a ) == ext_b );

        const auto faces_a = faceFunctions( la, comp_a, a.side_id );
        const auto faces_b = faceFunctions( lb, comp_b, b.side_id );

        std::vector<Constraint> out;
        for( const auto& [idx_a, fid_a] : faces_a )
        {
            Constraint c;
            c.side_key = makeSideKey( a, b );
            c.patch_a = a.constituent_id;
            c.local_a = fid_a;
            c.comp_a = comp_a;
            c.patch_b = b.constituent_id;
            c.local_b = faces_b.at( transformSideIndex( t, idx_a, ext_a ) );
            c.comp_b = comp_b;
            c.aligned = aligned;
            out.push_back( c );
        }
        return out;
    }

    // Parity of the 3d axis permutation a gluing induces: normal axes
    // correspond, tangential axes follow the side transform.  True when even.
    bool axisPermutationEven( const size_t side_a,
                              const size_t side_b,
                              const SideCoordinateTransform& t )
    {
        const std::vector<size_t> src_axes = tangentialAxes( 3, side_a );
        const std::vector<size_t> dst_axes = tangentialAxes( 3, side_b );
        std::array<size_t, 3> sigma{};
        sigma.at( side_a / 2 ) = side_b / 2;
        for( size_t d = 0; d < dst_axes.size(); d++ )
            sigma.at( src_axes.at( t.source_axis_for_destination.at( d ) ) ) = dst_axes.at( d );

        size_t inversions = 0;
        for( size_t i = 0; i < 3; i++ )
            for( size_t j = i + 1; j < 3; j++ )
                if( sigma.at( i ) > sigma.at( j ) ) inversions++;
        return inversions % 2 == 0;
    }

    // The intended constraints across one declared interface.  conforming_comps
    // names, per destination tangential position d, the pair of components that
    // must merge.  For H(curl) those are the two tangential components, paired
    // by the axis correspondence; for H(div) the single normal one.
    std::vector<Constraint> interfaceConstraints(
        const ConstituentSide& a,
        const TPPermutation perm,
        const ConstituentSide& b,
        const std::vector<ComponentLengths>& patch_lengths,
        const ConformingType conforming_type )
    {
        constexpr size_t dim = 3;
        const SideCoordinateTransform t = sideCoordinateTransform( dim, a.side_id, b.side_id, perm );
        const std::vector<size_t> src_axes = tangentialAxes( dim, a.side_id );
        const std::vector<size_t> dst_axes = tangentialAxes( dim, b.side_id );
        const ComponentLengths& la = patch_lengths.at( a.constituent_id );
        const ComponentLengths& lb = patch_lengths.at( b.constituent_id );

        // (comp_a, comp_b, expected relative sign).
        std::vector<std::array<size_t, 2>> comp_pairs;
        std::vector<bool> comp_aligned;
        if( conforming_type == ConformingType::Curl )
        {
            // De Rham tangential rule: the conforming components on a face are
            // exactly those whose component index is a tangential axis.  A
            // component along a reversed tangential axis changes sign, so the
            // expected relative sign is that axis's reversal flag.
            for( size_t d = 0; d < dst_axes.size(); d++ )
            {
                comp_pairs.push_back( { src_axes.at( t.source_axis_for_destination.at( d ) ), dst_axes.at( d ) } );
                comp_aligned.push_back( not t.source_axis_reversed.at( d ) );
            }
        }
        else
        {
            // De Rham normal rule: the one conforming component on a face is
            // the normal one.  Piola gives v_b^sigma(i) = s_i v_a^i / det J
            // with det J = sgn(sigma) * prod_i s_i, so at i = normal the sign
            // is sgn(sigma) times the TANGENTIAL reversals -- the normal flip
            // cancels.  Derived in M4; unlike H(curl) above, one component's
            // sign couples both tangential flips and the permutation parity.
            bool flip = not axisPermutationEven( a.side_id, b.side_id, t );
            for( size_t d = 0; d < t.source_axis_reversed.size(); d++ )
                if( t.source_axis_reversed.at( d ) ) flip = not flip;
            comp_pairs.push_back( { a.side_id / 2, b.side_id / 2 } );
            comp_aligned.push_back( not flip );
        }

        std::vector<Constraint> out;
        for( size_t c_ii = 0; c_ii < comp_pairs.size(); c_ii++ )
        {
            const auto one = pairFaceFunctions( a, t, b, la, lb, comp_pairs.at( c_ii ).at( 0 ),
                                                comp_pairs.at( c_ii ).at( 1 ), comp_aligned.at( c_ii ) );
            out.insert( out.end(), one.begin(), one.end() );
        }
        return out;
    }

    // The scalar analogue, used to validate the enumeration above against the
    // scalar multipatch merge before it is used to judge the vector merge.
    std::vector<Constraint> scalarConstraints( const ConstituentSide& a,
                                               const TPPermutation perm,
                                               const ConstituentSide& b,
                                               const std::vector<ComponentLengths>& patch_lengths )
    {
        const SideCoordinateTransform t = sideCoordinateTransform( 3, a.side_id, b.side_id, perm );
        return pairFaceFunctions( a, t, b, patch_lengths.at( a.constituent_id ),
                                  patch_lengths.at( b.constituent_id ), 0, 0, true );
    }
}

namespace
{
    // What the comparison against production measured.
    struct MergeObs
    {
        size_t n_local = 0;        // summed patch-local functions
        size_t n_global = 0;       // production's global count
        size_t n_global_expected = 0; // the independent closure's count
        size_t n_constraints = 0;
        size_t n_misaligned = 0;   // constraints whose expected sign is a flip
        // Both mismatch counts below are per LOCAL ENTRY, and id_mismatch is
        // additionally per direction, so neither is a count of classes.
        size_t id_mismatch = 0;
        size_t sign_mismatch = 0;
        size_t sign_mismatch_classes = 0; // distinct classes holding a disagreement
    };

    std::vector<size_t> patchOffsets( const std::vector<ComponentLengths>& patch_lengths )
    {
        std::vector<size_t> offsets{ 0 };
        for( const ComponentLengths& cl : patch_lengths )
        {
            size_t n = 0;
            for( const std::vector<size_t>& lens : cl ) n += blockSize( lens );
            offsets.push_back( offsets.back() + n );
        }
        return offsets;
    }

    // Per-constraint id and sign comparison (M4 assertions 3 and 4), then the
    // partition as a whole.  The whole-partition part closes the independent
    // constraint set with a union-find and compares the resulting equivalence
    // relation to production's, in BOTH directions, so over-merging and
    // under-merging are both visible.  util::UnionFind is reused there only as
    // a generic transitive-closure utility; the quantity under test is the
    // constraint set, which is derived independently.
    MergeObs compareMerge( const VectorConformingMultiPatchSplineSpace& space,
                           const std::vector<ComponentLengths>& patch_lengths,
                           const std::vector<Constraint>& constraints )
    {
        MergeObs o;
        const std::vector<size_t> offsets = patchOffsets( patch_lengths );
        o.n_local = offsets.back();
        o.n_global = space.numFunctions();
        o.n_constraints = constraints.size();

        const auto& fid_map = space.functionIdMap();

        // M4 assertions 3 and 4, one check per intended face-level pairing.
        for( const Constraint& c : constraints )
        {
            if( not c.aligned ) o.n_misaligned++;
            const auto& [gid_a, or_a] = fid_map.at( c.patch_a ).at( c.local_a );
            const auto& [gid_b, or_b] = fid_map.at( c.patch_b ).at( c.local_b );
            CHECK( gid_a == gid_b );
            CHECK( ( or_a == or_b ) == c.aligned );
        }

        util::UnionFind uf( o.n_local );
        for( const Constraint& c : constraints )
        {
            REQUIRE_NOTHROW( uf.unite( offsets.at( c.patch_a ) + c.local_a,
                                       offsets.at( c.patch_b ) + c.local_b,
                                       c.aligned ) );
        }
        o.n_global_expected = uf.numSets();

        // Partition equality.  class_gauge is the per-class sign offset between
        // the two conventions; it must be constant on a class exactly when the
        // relative signs agree throughout it.
        std::map<size_t, size_t> root_to_gid;
        std::map<size_t, size_t> gid_to_root;
        std::map<size_t, bool> class_gauge;
        std::set<size_t> sign_bad_roots;
        for( size_t patch_ii = 0; patch_ii < patch_lengths.size(); patch_ii++ )
        {
            for( size_t local_ii = 0; local_ii < fid_map.at( patch_ii ).size(); local_ii++ )
            {
                const auto& [gid, orientation] = fid_map.at( patch_ii ).at( local_ii );
                const auto [root, parity] = uf.findWithOrientation( offsets.at( patch_ii ) + local_ii );
                const size_t gid_val = static_cast<size_t>( gid.id() );

                root_to_gid.try_emplace( root, gid_val );
                gid_to_root.try_emplace( gid_val, root );
                class_gauge.try_emplace( root, parity != orientation );

                if( root_to_gid.at( root ) != gid_val ) o.id_mismatch++;
                if( gid_to_root.at( gid_val ) != root ) o.id_mismatch++;
                if( class_gauge.at( root ) != ( parity != orientation ) )
                {
                    o.sign_mismatch++;
                    sign_bad_roots.insert( root );
                }
            }
        }
        o.sign_mismatch_classes = sign_bad_roots.size();
        return o;
    }

    // The declared interfaces, one entry per unordered side pair.
    std::vector<std::tuple<ConstituentSide, TPPermutation, ConstituentSide>>
        declaredInterfaces( const MultiPatchCombinatorialMap& mp )
    {
        std::vector<std::tuple<ConstituentSide, TPPermutation, ConstituentSide>> out;
        for( const auto& [first, connection] : mp.connections() )
        {
            const auto& [permutation, second] = connection;
            if( not( first < second ) ) continue;
            out.push_back( { first, permutation, second } );
        }
        return out;
    }

    MultiPatchSplineSpace buildPrimalFromDarts( const size_t n_patches, const DartConnections& conns )
    {
        const KnotVector kv( { 0, 0, 0, 1, 1, 1 }, ptol );
        const auto ss_tp = std::make_shared<const TPSplineSpace>( buildBSpline( { kv, kv, kv }, { 2, 2, 2 } ) );
        return buildMultiPatchSplineSpace(
            std::vector<std::shared_ptr<const TPSplineSpace>>( n_patches, ss_tp ), conns );
    }

    MultiPatchSplineSpace buildPrimalFromSides( const size_t n_patches, const InternalConnectionsMap& conns )
    {
        const KnotVector kv( { 0, 0, 0, 1, 1, 1 }, ptol );
        const auto ss_tp = std::make_shared<const TPSplineSpace>( buildBSpline( { kv, kv, kv }, { 2, 2, 2 } ) );
        return buildMultiPatchSplineSpace(
            std::vector<std::shared_ptr<const TPSplineSpace>>( n_patches, ss_tp ), conns );
    }

    std::vector<ComponentLengths> lengthsOf( const VectorConformingMultiPatchSplineSpace& space )
    {
        std::vector<ComponentLengths> out;
        for( const auto& patch : space.subSpaces() ) out.push_back( componentLengths( *patch ) );
        return out;
    }

    void reportMerge( const std::string& label, const MergeObs& o )
    {
        std::cout << "  " << label << ": local " << o.n_local << ", global " << o.n_global << " (expected "
                  << o.n_global_expected << "), constraints " << o.n_constraints << " of which "
                  << o.n_misaligned << " sign-flipping; id mismatch entries " << o.id_mismatch
                  << ", sign mismatch entries " << o.sign_mismatch << " in "
                  << o.sign_mismatch_classes << " classes" << std::endl;
    }
}

namespace
{
    // M4's graph G itself: vertices are (patch, local function), edges are the
    // generated constraints.  An edge lying on a CYCLE of G is a co-tree edge
    // under some spanning forest, so it is an edge whose parity test is the
    // non-vacuous one; a BRIDGE carries no such test.  This is the quantity M4
    // assertion 6 asks about.  Keying on the interface SideKey instead would
    // only show that a reversing INTERFACE touches a cyclic topological edge,
    // which does not imply that any particular DOF constraint of that
    // interface lies on a cycle of G -- most are face-interior.
    struct CycleObs
    {
        size_t n_edges = 0;
        size_t n_nonbridge = 0;      // edges of G lying on some cycle
        size_t n_flip_nonbridge = 0; // sign-flipping edges lying on some cycle
        size_t max_patches = 0;      // patches in the richest such component
        size_t max_sides = 0;        // declared interfaces in that component
    };

    CycleObs dofCycleObs( const std::vector<Constraint>& constraints, const std::vector<size_t>& offsets )
    {
        constexpr size_t none = std::numeric_limits<size_t>::max();
        const size_t n_v = offsets.back();

        std::vector<std::vector<std::pair<size_t, size_t>>> adj( n_v ); // (neighbour, edge id)
        std::vector<std::array<size_t, 2>> ends;
        for( const Constraint& c : constraints )
        {
            const size_t u = offsets.at( c.patch_a ) + c.local_a;
            const size_t v = offsets.at( c.patch_b ) + c.local_b;
            adj.at( u ).push_back( { v, ends.size() } );
            adj.at( v ).push_back( { u, ends.size() } );
            ends.push_back( { u, v } );
        }

        // Tarjan low-link.  The traversed EDGE is skipped rather than the
        // parent vertex, so parallel constraints correctly register a cycle.
        std::vector<size_t> disc( n_v, none ), low( n_v, none ), comp( n_v, none );
        std::vector<bool> is_bridge( ends.size(), false );
        size_t timer = 0, n_comp = 0;
        for( size_t root = 0; root < n_v; root++ )
        {
            if( disc.at( root ) != none ) continue;
            const size_t this_comp = n_comp++;
            disc.at( root ) = low.at( root ) = timer++;
            comp.at( root ) = this_comp;
            // Explicit stack of (vertex, entry edge, next adjacency index).
            std::vector<std::array<size_t, 3>> stack{ { root, none, 0 } };
            while( not stack.empty() )
            {
                const size_t top = stack.size() - 1;
                const size_t u = stack.at( top ).at( 0 );
                const size_t in_edge = stack.at( top ).at( 1 );
                const size_t ii = stack.at( top ).at( 2 );
                if( ii < adj.at( u ).size() )
                {
                    stack.at( top ).at( 2 ) = ii + 1;
                    const auto [v, e] = adj.at( u ).at( ii );
                    if( e == in_edge ) continue;
                    if( disc.at( v ) == none )
                    {
                        disc.at( v ) = low.at( v ) = timer++;
                        comp.at( v ) = this_comp;
                        stack.push_back( { v, e, 0 } );
                    }
                    else low.at( u ) = std::min( low.at( u ), disc.at( v ) );
                }
                else
                {
                    stack.pop_back();
                    if( stack.empty() ) continue;
                    const size_t parent = stack.back().at( 0 );
                    low.at( parent ) = std::min( low.at( parent ), low.at( u ) );
                    if( low.at( u ) > disc.at( parent ) ) is_bridge.at( in_edge ) = true;
                }
            }
        }

        // Which patch each vertex belongs to, and what each component holds.
        std::map<size_t, std::set<size_t>> comp_patches;
        for( size_t v = 0; v < n_v; v++ )
        {
            if( comp.at( v ) == none ) continue;
            const size_t patch =
                std::upper_bound( offsets.begin(), offsets.end(), v ) - offsets.begin() - 1;
            comp_patches[comp.at( v )].insert( patch );
        }
        std::map<size_t, std::set<SideKey>> comp_sides;
        for( size_t e = 0; e < ends.size(); e++ )
            comp_sides[comp.at( ends.at( e ).at( 0 ) )].insert( constraints.at( e ).side_key );

        CycleObs o;
        o.n_edges = ends.size();
        for( size_t e = 0; e < ends.size(); e++ )
        {
            if( is_bridge.at( e ) ) continue;
            o.n_nonbridge++;
            if( constraints.at( e ).aligned ) continue;
            o.n_flip_nonbridge++;
            const size_t c = comp.at( ends.at( e ).at( 0 ) );
            o.max_patches = std::max( o.max_patches, comp_patches.at( c ).size() );
            o.max_sides = std::max( o.max_sides, comp_sides.at( c ).size() );
        }
        return o;
    }

    CycleObs curlCycles( const MultiPatchSplineSpace& primal, const MultiPatchCombinatorialMap& mp )
    {
        const auto space = buildHCurlMultiPatchSplineSpace( primal );
        const std::vector<ComponentLengths> lengths = lengthsOf( space );
        std::vector<Constraint> constraints;
        for( const auto& [a, perm, b] : declaredInterfaces( mp ) )
        {
            const auto one = interfaceConstraints( a, perm, b, lengths, ConformingType::Curl );
            constraints.insert( constraints.end(), one.begin(), one.end() );
        }
        return dofCycleObs( constraints, patchOffsets( lengths ) );
    }

    void reportCycles( const CycleObs& o )
    {
        std::cout << "  G: " << o.n_edges << " constraint edges, " << o.n_nonbridge
                  << " on cycles, of which sign-flipping " << o.n_flip_nonbridge;
        if( o.n_flip_nonbridge > 0 )
            std::cout << "; richest such component spans " << o.max_patches << " patches and "
                      << o.max_sides << " interfaces";
        std::cout << std::endl;
    }

    // Everything step 6 asserts for one fixture and one conforming type.
    MergeObs runStep6( const std::string& label,
                       const MultiPatchSplineSpace& primal,
                       const MultiPatchCombinatorialMap& mp,
                       const ConformingType conforming_type )
    {
        const VectorConformingMultiPatchSplineSpace space =
            conforming_type == ConformingType::Curl ? buildHCurlMultiPatchSplineSpace( primal )
                                                    : buildHDivMultiPatchSplineSpace( primal );
        const std::vector<ComponentLengths> patch_lengths = lengthsOf( space );

        std::vector<Constraint> constraints;
        for( const auto& [a, perm, b] : declaredInterfaces( mp ) )
        {
            const auto one = interfaceConstraints( a, perm, b, patch_lengths, conforming_type );
            constraints.insert( constraints.end(), one.begin(), one.end() );
        }

        // Both conforming types now carry an independently derived sign: the
        // covariant tangential rule for H(curl), the contravariant normal rule
        // for H(div).  Neither reads coordinateTransform or functionIdMap.
        const MergeObs o = compareMerge( space, patch_lengths, constraints );
        reportMerge( label, o );
        CHECK( o.id_mismatch == 0 );
        CHECK( o.n_global == o.n_global_expected );
        CHECK( o.sign_mismatch == 0 );
        return o;
    }
}

TEST_CASE( "STEP 6: H(curl) face constraints on the four-patch 3d fixture", "[rowd]" )
{
    const DartConnections declared = fourPatchConnections();
    const auto mp = buildMultiPatch3d( 4, declared );
    const MultiPatchSplineSpace primal = buildPrimalFromDarts( 4, declared );

    std::cout << "STEP 6: four-patch 3d fixture" << std::endl;

    const auto curl_space = buildHCurlMultiPatchSplineSpace( primal );
    const auto lengths = lengthsOf( curl_space );
    std::cout << "  H(curl) per-component per-axis function counts, patch 0:" << std::endl;
    for( size_t k = 0; k < lengths.at( 0 ).size(); k++ )
    {
        std::cout << "    component " << k << ":";
        for( const size_t n : lengths.at( 0 ).at( k ) ) std::cout << " " << n;
        std::cout << std::endl;
    }

    const MergeObs curl = runStep6( "H(curl)", primal, *mp, ConformingType::Curl );
    const MergeObs div = runStep6( "H(div) control", primal, *mp, ConformingType::Divergence );

    // STEP 7.  The contravariant face sign is now predicted, not just reported:
    // 12 of this fixture's 24 normal-component constraints are expected to
    // flip.  H(curl) flips nowhere here, so these come entirely from the
    // permutation parity -- the two rules are independent.
    CHECK( div.n_constraints == 24 );
    CHECK( div.n_misaligned == 12 );

    const CycleObs cyc = curlCycles( primal, *mp );
    reportCycles( cyc );

    // G has co-tree edges here, so the union-find's parity test is not vacuous
    // on this fixture for structural reasons.  What is missing is a nonzero
    // cochain to test it with; see below.
    CHECK( cyc.n_nonbridge > 0 );

    // PERMANENT characterization of THIS fixture, not a defect: every declared
    // interface here has source_axis_reversed == {0,0}, so every expected
    // H(curl) relative sign is "aligned".  Assertions 4 and 6 are therefore
    // vacuous on this fixture; assertion 4 is discriminating on the reversing
    // four-patch fixture below, assertion 6 only on the reversing ring.  These
    // zeros guard against a future edit silently making this fixture reversing
    // and so changing what the sign-trivial baseline measures.
    CHECK( cyc.n_flip_nonbridge == 0 );
    CHECK( curl.n_misaligned == 0 );
}

TEST_CASE( "STEP 6: H(curl) face constraints on the swept three-patch 3d fixture", "[rowd]" )
{
    const InternalConnectionsMap conns = connectionsOfSweptMultipatch( threePatchRing2d() );
    const auto mp = buildMultiPatch3dFromSides( 3, conns );
    const MultiPatchSplineSpace primal = buildPrimalFromSides( 3, conns );

    std::cout << "STEP 6: swept three-patch 3d fixture" << std::endl;

    const MergeObs curl = runStep6( "H(curl)", primal, *mp, ConformingType::Curl );
    const MergeObs div = runStep6( "H(div) control", primal, *mp, ConformingType::Divergence );

    // STEP 7: all 12 normal-component constraints are expected to flip here.
    CHECK( div.n_constraints == 12 );
    CHECK( div.n_misaligned == 12 );

    const CycleObs cyc = curlCycles( primal, *mp );
    reportCycles( cyc );
    CHECK( cyc.n_nonbridge > 0 );

    // PERMANENT, as above: this fixture is sign-trivial too.
    CHECK( cyc.n_flip_nonbridge == 0 );
    CHECK( curl.n_misaligned == 0 );
}

namespace
{
    // Oracle self-check.  The same enumeration, index transform and local-id
    // layout, applied to the scalar H1 space, must reproduce the scalar
    // multipatch merge exactly.  MultiPatchSplineSpace_test's test_c0 certifies
    // that merge by evaluation for FIXTURE A's pinned 65-function case ONLY;
    // agreement on B, C and D is cross-implementation consistency, which is
    // weaker.  A disagreement here would localise the fault in this file's
    // enumeration rather than in the vector merge.
    struct ScalarObs
    {
        size_t n_local = 0;
        size_t n_global = 0;
        size_t n_global_expected = 0;
        size_t n_constraints = 0;
        size_t id_mismatch = 0;
    };

    ScalarObs compareScalarMerge( const MultiPatchSplineSpace& primal, const MultiPatchCombinatorialMap& mp )
    {
        // One "component" per patch, the full-degree scalar basis.
        std::vector<ComponentLengths> patch_lengths;
        for( const auto& patch : primal.subSpaces() )
        {
            std::vector<size_t> lengths;
            for( const auto& basis_1d : tensorProductComponentSplines( *patch ) )
                lengths.push_back( basis_1d->numFunctions() );
            patch_lengths.push_back( ComponentLengths{ lengths } );
        }

        std::vector<Constraint> constraints;
        for( const auto& [a, perm, b] : declaredInterfaces( mp ) )
        {
            const auto one = scalarConstraints( a, perm, b, patch_lengths );
            constraints.insert( constraints.end(), one.begin(), one.end() );
        }

        ScalarObs o;
        const std::vector<size_t> offsets = patchOffsets( patch_lengths );
        o.n_local = offsets.back();
        o.n_global = primal.numFunctions();
        o.n_constraints = constraints.size();

        util::UnionFind uf( o.n_local );
        for( const Constraint& c : constraints )
            uf.unite( offsets.at( c.patch_a ) + c.local_a, offsets.at( c.patch_b ) + c.local_b, true );
        o.n_global_expected = uf.numSets();

        const auto& fid_map = primal.functionIdMap();
        std::map<size_t, size_t> root_to_gid, gid_to_root;
        for( size_t patch_ii = 0; patch_ii < patch_lengths.size(); patch_ii++ )
        {
            for( size_t local_ii = 0; local_ii < fid_map.at( patch_ii ).size(); local_ii++ )
            {
                const size_t gid = static_cast<size_t>( fid_map.at( patch_ii ).at( local_ii ).id() );
                const auto [root, parity] = uf.findWithOrientation( offsets.at( patch_ii ) + local_ii );
                (void)parity;
                root_to_gid.try_emplace( root, gid );
                gid_to_root.try_emplace( gid, root );
                if( root_to_gid.at( root ) != gid ) o.id_mismatch++;
                if( gid_to_root.at( gid ) != root ) o.id_mismatch++;
            }
        }
        return o;
    }

    void reportTransforms( const MultiPatchCombinatorialMap& mp )
    {
        for( const auto& [a, perm, b] : declaredInterfaces( mp ) )
        {
            const SideCoordinateTransform t = sideCoordinateTransform( 3, a.side_id, b.side_id, perm );
            std::cout << "    " << a << " -> " << b << "  " << perm << "  src_for_dst = {"
                      << t.source_axis_for_destination.at( 0 ) << "," << t.source_axis_for_destination.at( 1 )
                      << "}  reversed = {" << t.source_axis_reversed.at( 0 ) << ","
                      << t.source_axis_reversed.at( 1 ) << "}" << std::endl;
        }
    }
}

TEST_CASE( "STEP 6 PRECONDITION: the face enumeration reproduces the scalar merge", "[rowd]" )
{
    std::cout << "STEP 6 PRECONDITION: scalar merge agreement" << std::endl;

    const DartConnections declared = fourPatchConnections();
    const auto mp4 = buildMultiPatch3d( 4, declared );
    const MultiPatchSplineSpace primal4 = buildPrimalFromDarts( 4, declared );
    std::cout << "  four-patch interface transforms:" << std::endl;
    reportTransforms( *mp4 );
    const ScalarObs s4 = compareScalarMerge( primal4, *mp4 );
    std::cout << "  four-patch: local " << s4.n_local << ", global " << s4.n_global << " (expected "
              << s4.n_global_expected << "), constraints " << s4.n_constraints << ", id mismatches "
              << s4.id_mismatch << std::endl;

    const InternalConnectionsMap conns = connectionsOfSweptMultipatch( threePatchRing2d() );
    const auto mp3 = buildMultiPatch3dFromSides( 3, conns );
    const MultiPatchSplineSpace primal3 = buildPrimalFromSides( 3, conns );
    std::cout << "  swept three-patch interface transforms:" << std::endl;
    reportTransforms( *mp3 );
    const ScalarObs s3 = compareScalarMerge( primal3, *mp3 );
    std::cout << "  swept three-patch: local " << s3.n_local << ", global " << s3.n_global << " (expected "
              << s3.n_global_expected << "), constraints " << s3.n_constraints << ", id mismatches "
              << s3.id_mismatch << std::endl;

    const DartConnections rev_declared = reversingFourPatchConnections();
    const auto mpr = buildMultiPatch3d( 4, rev_declared );
    const MultiPatchSplineSpace primalr = buildPrimalFromDarts( 4, rev_declared );
    std::cout << "  reversing four-patch interface transforms:" << std::endl;
    reportTransforms( *mpr );
    const ScalarObs sr = compareScalarMerge( primalr, *mpr );
    std::cout << "  reversing four-patch: local " << sr.n_local << ", global " << sr.n_global
              << " (expected " << sr.n_global_expected << "), constraints " << sr.n_constraints
              << ", id mismatches " << sr.id_mismatch << std::endl;

    const DartConnections ring_declared = reversingRingConnections();
    const auto mpring = buildMultiPatch3d( 3, ring_declared );
    const MultiPatchSplineSpace primalring = buildPrimalFromDarts( 3, ring_declared );
    std::cout << "  reversing three-patch ring interface transforms:" << std::endl;
    reportTransforms( *mpring );
    const ScalarObs sring = compareScalarMerge( primalring, *mpring );
    std::cout << "  reversing ring: local " << sring.n_local << ", global " << sring.n_global
              << " (expected " << sring.n_global_expected << "), constraints " << sring.n_constraints
              << ", id mismatches " << sring.id_mismatch << std::endl;

    CHECK( sring.n_local == 3 * 27 );
    CHECK( sring.n_global_expected == sring.n_global );
    CHECK( sring.id_mismatch == 0 );

    // 65 is the value test_c0 pins in MultiPatchSplineSpace_test.cpp:353.
    CHECK( s4.n_local == 4 * 27 );
    CHECK( s4.n_global == 65 );
    CHECK( s4.n_global_expected == s4.n_global );
    CHECK( s4.id_mismatch == 0 );

    CHECK( sr.n_local == 4 * 27 );
    CHECK( sr.n_global_expected == sr.n_global );
    CHECK( sr.id_mismatch == 0 );

    CHECK( s3.n_local == 3 * 27 );
    CHECK( s3.n_global_expected == s3.n_global );
    CHECK( s3.id_mismatch == 0 );
}

TEST_CASE( "STEP 6: H(curl) face constraints on the reversing four-patch 3d fixture", "[rowd]" )
{
    const DartConnections declared = reversingFourPatchConnections();
    const auto mp = buildMultiPatch3d( 4, declared );
    const MultiPatchSplineSpace primal = buildPrimalFromDarts( 4, declared );

    std::cout << "STEP 6: reversing four-patch 3d fixture" << std::endl;
    reportTransforms( *mp );

    CHECK( mp->connections().size() == 2 * declared.size() );

    const MergeObs curl = runStep6( "H(curl)", primal, *mp, ConformingType::Curl );
    const MergeObs div = runStep6( "H(div) control", primal, *mp, ConformingType::Divergence );

    // STEP 7: 16 of 24, up from fixture A's 12, because this fixture's extra
    // tangential reversal flips the normal component on the re-glued interface
    // as well.
    CHECK( div.n_constraints == 24 );
    CHECK( div.n_misaligned == 16 );

    const CycleObs cyc = curlCycles( primal, *mp );
    reportCycles( cyc );

    // M4 assertion 4 is discriminating here: some expected relative signs are
    // flips, so runStep6's sign comparison can distinguish correct sign
    // handling from unconditional alignment.
    CHECK( curl.n_misaligned > 0 );

    // G has cycles here, but ASSERTION 6 IS NOT MET ON THIS FIXTURE, and the
    // zero below is a measured structural fact rather than a defect.  This
    // fixture reverses a tangential axis that is NOT the direction of any edge
    // junction it carries, so all six of its flips are BRIDGES of G.  An
    // interface-level test would wrongly read 6 here; see the comment on
    // reversingRingConnections for why only the edge-direction component can
    // lie on a cycle, and the ring fixture below for assertion 6 proper.
    CHECK( cyc.n_nonbridge > 0 );
    CHECK( cyc.n_flip_nonbridge == 0 );
}

TEST_CASE( "STEP 6: H(curl) face constraints on the reversing three-patch ring", "[rowd]" )
{
    const DartConnections declared = reversingRingConnections();
    const auto mp = buildMultiPatch3d( 3, declared );
    const MultiPatchSplineSpace primal = buildPrimalFromDarts( 3, declared );

    std::cout << "STEP 6: reversing three-patch ring" << std::endl;
    reportTransforms( *mp );

    CHECK( mp->connections().size() == 2 * declared.size() );

    const MergeObs curl = runStep6( "H(curl)", primal, *mp, ConformingType::Curl );
    const MergeObs div = runStep6( "H(div) control", primal, *mp, ConformingType::Divergence );

    // STEP 7: 4 of 12.  Contrast H(curl)'s 24 of 36 flips on this same fixture:
    // the two rules disagree about which interfaces reverse, which is why
    // H(div) is a control on the covariant result rather than a restatement.
    CHECK( div.n_constraints == 12 );
    CHECK( div.n_misaligned == 4 );

    const CycleObs cyc = curlCycles( primal, *mp );
    reportCycles( cyc );

    // Assertion 4, discriminating: some expected relative signs are flips.
    CHECK( curl.n_misaligned > 0 );

    // M4 ASSERTION 6, stated on G itself: at least one SIGN-FLIPPING constraint
    // is a NON-BRIDGE edge of the DOF-constraint graph.  Only then is the
    // parity cochain nonzero on some cycle, so the union-find's co-tree test is
    // exercised against a nontrivial class rather than satisfied by the zero
    // gauge.  Interface-level cyclicity does NOT imply this: a reversing
    // interface's constraints are mostly face-interior and lie on no cycle.
    CHECK( cyc.n_nonbridge > 0 );
    CHECK( cyc.n_flip_nonbridge > 0 );

    // And that cycle is a genuine edge junction rather than a two-patch face:
    // its component of G spans at least three patches and three interfaces.
    CHECK( cyc.max_patches >= 3 );
    CHECK( cyc.max_sides >= 3 );
}

namespace
{
    // STEP 8 (M5).  coordinateTransform in ParametricAtlas.cpp and
    // sideCoordinateTransform express the same gluing in two different
    // vocabularies, and M5 is open because only the second has been derived.
    // Reconciling them by fitting would prove nothing, so the expectation
    // below is derived from sideCoordinateTransform plus the geometry of
    // getFrame, and coordinateTransform's own formula is not consulted.
    //
    // THE TWO VOCABULARIES.  sideCoordinateTransform is a signed permutation
    // of TANGENTIAL axes indexed by DESTINATION tangential axis.
    // coordinateTransform is indexed by SOURCE PARENT axis and holds
    // (destination parent axis, aligned) for all dim axes, normal included.
    //
    // THE TANGENTIAL BLOCK.  getFrame is called with reverse_dart = true on
    // the far side, which swaps ppt00 and ppt10, so the two frames take the
    // same two physical points as origin and first corner.  Their tangential
    // frame directions are therefore the same physical directions, and the
    // index correspondence reverses exactly when the two patches' parent-axis
    // senses disagree: aligned = not source_axis_reversed[d].
    //
    // THE NORMAL BLOCK, which is where the unexplained factor term lived.  In
    // 3d getFrame reaches its fourth corner by phi {2,1,1}, which lands OFF
    // the face and INWARD into the patch, so each frame's third direction
    // points inward and the two point physically OPPOSITE ways.  Inward is
    // +e_n exactly on the LOWER side of an axis, and the side ordering is
    // S1,S0,T1,T0,U1,U0, so lower means an ODD side id.  A transition map
    // sends outward-a to inward-b, giving normal sign -sigma_a*sigma_b:
    // aligned iff the two side ids have DIFFERENT parity.  That extra flip is
    // exactly what factor = ( coord != dim - 1 ) supplies.
    std::map<size_t, std::pair<size_t, bool>>
        expectedCoordinateTransform( const size_t side_a, const size_t side_b, const SideCoordinateTransform& t )
    {
        const std::vector<size_t> src_axes = tangentialAxes( 3, side_a );
        const std::vector<size_t> dst_axes = tangentialAxes( 3, side_b );
        std::map<size_t, std::pair<size_t, bool>> expected;
        for( size_t d = 0; d < dst_axes.size(); d++ )
            expected[src_axes.at( t.source_axis_for_destination.at( d ) )] = {
                dst_axes.at( d ), not t.source_axis_reversed.at( d ) };
        expected[side_a / 2] = { side_b / 2, ( side_a % 2 ) != ( side_b % 2 ) };
        return expected;
    }

    // Every (source side, destination side, permutation) class realisable
    // between two 3d patches, with one representative dart pair each.  A
    // 1-element hex has all 24 of its darts on the boundary, 4 per side, and
    // the dart-pair constructor DERIVES the permutation, so sweeping 24x24
    // enumerates the classes rather than assuming which are reachable.
    using GluingClass = std::tuple<size_t, size_t, size_t>;

    std::map<GluingClass, DartConnections> gluingClasses3d( size_t& n_built, size_t& n_rejected )
    {
        std::map<GluingClass, DartConnections> out;
        n_built = 0;
        n_rejected = 0;
        for( Dart::IndexType da = 0; da < 24; da++ )
        {
            for( Dart::IndexType db = 0; db < 24; db++ )
            {
                const DartConnections conns{ { { 0, Dart( da ) }, { 1, Dart( db ) } } };
                std::shared_ptr<const MultiPatchCombinatorialMap> mp;
                try
                {
                    mp = buildMultiPatch3d( 2, conns );
                }
                catch( const std::exception& )
                {
                    n_rejected++;
                    continue;
                }
                n_built++;
                for( const auto& [from, to] : mp->connections() )
                {
                    if( from.constituent_id != 0 ) continue;
                    out.insert( { { from.side_id, to.second.side_id, static_cast<size_t>( to.first ) }, conns } );
                }
            }
        }
        return out;
    }
}

TEST_CASE( "STEP 8: coordinateTransform against sideCoordinateTransform, all 3d gluing classes", "[rowd]" )
{
    const KnotVector kv( { 0, 0, 0, 1, 1, 1 }, ptol );
    const auto cmap_1d = std::make_shared<const CombinatorialMap1d>( numElements( kv ) );
    const auto param_1d = std::make_shared<const param::ParametricAtlas1d>( cmap_1d, parametricLengths( kv ) );
    const auto cmap_2d = std::make_shared<const TPCombinatorialMap>( cmap_1d, cmap_1d );
    const auto param_2d = std::make_shared<const param::TPParametricAtlas>( cmap_2d, param_1d, param_1d );
    const auto cmap_3d = std::make_shared<const TPCombinatorialMap>( cmap_2d, cmap_1d );
    const auto param_3d = std::make_shared<const param::TPParametricAtlas>( cmap_3d, param_2d, param_1d );

    size_t n_built = 0, n_rejected = 0;
    const auto classes = gluingClasses3d( n_built, n_rejected );
    std::cout << "STEP 8: dart pairs built " << n_built << ", rejected " << n_rejected
              << ", distinct gluing classes " << classes.size() << std::endl;

    // 6 sides x 6 sides x 4 permutations, every one of them reachable from a
    // dart pair.  Asserted after measuring, so the exhaustiveness M5 asks for
    // is a measured fact about this enumeration and not an assumption in it.
    CHECK( n_built == 576 );
    CHECK( n_rejected == 0 );
    CHECK( classes.size() == 144 );

    size_t n_dart_disagreements = 0, n_bad_dart_counts = 0, n_orientation_reversing = 0;
    for( const auto& [cls, conns] : classes )
    {
        const auto [side_a, side_b, perm_i] = cls;
        const auto mp = buildMultiPatch3d( 2, conns );
        const param::MultiPatchParametricAtlas atlas(
            mp, std::vector<std::shared_ptr<const param::TPParametricAtlas>>{ param_3d, param_3d } );

        const SideCoordinateTransform t = sideCoordinateTransform(
            3, side_a, side_b, static_cast<TPPermutation>( perm_i ) );
        const auto expected = expectedCoordinateTransform( side_a, side_b, t );

        // THE NORMAL ENTRY HAS TWO DERIVATIONS OF INDEPENDENT PROVENANCE:
        // step 7's phi-free contravariant sign sgn(sigma)*s_t1*s_t2, and the
        // parity formula above, s_n, from getFrame's inward corner.  Both are
        // compared against production, which reads this entry's aligned flag
        // straight into uf.unite for H(div) (VectorConformingMultiPatch-
        // SplineSpace.cpp:106,143).  Since det J = sgn(sigma) * prod_i s_i,
        // their agreement is ALGEBRAICALLY EQUIVALENT to det J = +1, so the
        // check below is an INTERPRETATION of an agreement already measured,
        // not a third confirmation of it.  What it says is that gluing two
        // parent cubes along a face is orientation PRESERVING on every
        // reachable class, verified here by enumeration.  Proving it without
        // enumeration reduces to one premise about the dart-pair constructor;
        // see M5 STEP 9.
        bool det_positive = axisPermutationEven( side_a, side_b, t );
        if( not expected.at( side_a / 2 ).second ) det_positive = not det_positive;
        for( size_t d = 0; d < t.source_axis_reversed.size(); d++ )
            if( t.source_axis_reversed.at( d ) ) det_positive = not det_positive;
        if( not det_positive ) n_orientation_reversing++;

        // A face cell has four darts.  The transform is a signed permutation
        // of parent axes, a representative-independent object, so all four
        // must agree; this also pins that no entry is left unwritten, the
        // failure mode a face-diagonal fourth corner would have produced.
        std::optional<SmallVector<std::pair<size_t, bool>, 3>> got;
        size_t n_darts = 0;
        for( Dart::IndexType local_d = 0; local_d < 24; local_d++ )
        {
            const Dart gd = mp->toGlobalDart( 0, Dart( local_d ) );
            if( not phi( *mp, 3, gd ).has_value() ) continue;
            n_darts++;
            const auto one = param::coordinateTransform( atlas, Face( gd ) );
            if( not got.has_value() ) got = one;
            else if( not( one == got.value() ) ) n_dart_disagreements++;
        }
        if( n_darts != 4 ) n_bad_dart_counts++;

        REQUIRE( got.has_value() );
        REQUIRE( got.value().size() == 3 );
        CHECK( std::set<size_t>{ got.value().at( 0 ).first,
                                 got.value().at( 1 ).first,
                                 got.value().at( 2 ).first }.size() == 3 );
        for( const auto& [src_axis, exp] : expected )
        {
            CHECK( got.value().at( src_axis ).first == exp.first );
            CHECK( got.value().at( src_axis ).second == exp.second );
        }
    }
    CHECK( n_dart_disagreements == 0 );
    CHECK( n_bad_dart_counts == 0 );
    std::cout << "  orientation-reversing classes: " << n_orientation_reversing << " of "
              << classes.size() << std::endl;
    CHECK( n_orientation_reversing == 0 );
}

// Coverage extension for steps 6 and 7 (M5 item i).  Those steps measured the
// covariant and contravariant sign rules only on the classes fixtures A-D
// happen to realise; this runs both over all 144.
//
// WHAT A TWO-PATCH SWEEP CAN AND CANNOT TEST.  Its constraint graph is a tree,
// so beta_1 = 0 and nothing here tests cycle consistency; that stays the
// contribution of the reversing fixtures C and D.  The per-constraint sign
// check is still discriminating, because the only gauge freedom is flipping a
// whole equivalence class, which flips both of a constraint's orientations
// together and so leaves the RELATIVE orientation it compares invariant.
TEST_CASE( "STEP 8: the step 6 and 7 interface sign rules over all 3d gluing classes", "[rowd]" )
{
    size_t n_built = 0, n_rejected = 0;
    const auto classes = gluingClasses3d( n_built, n_rejected );
    REQUIRE( classes.size() == 144 );

    struct Agg
    {
        size_t n_spaces = 0;
        size_t n_constraints = 0;
        size_t n_misaligned = 0;
        size_t n_classes_flipping = 0;
    };
    std::map<ConformingType, Agg> agg;
    size_t n_disagree = 0; // classes where the two rules differ on whether anything flips

    for( const auto& [cls, conns] : classes )
    {
        const auto mp = buildMultiPatch3d( 2, conns );
        const MultiPatchSplineSpace primal = buildPrimalFromDarts( 2, conns );
        std::map<ConformingType, bool> flips;
        for( const ConformingType ct : { ConformingType::Curl, ConformingType::Divergence } )
        {
            const VectorConformingMultiPatchSplineSpace space =
                ct == ConformingType::Curl ? buildHCurlMultiPatchSplineSpace( primal )
                                           : buildHDivMultiPatchSplineSpace( primal );
            const std::vector<ComponentLengths> lengths = lengthsOf( space );
            std::vector<Constraint> constraints;
            for( const auto& [a, perm, b] : declaredInterfaces( *mp ) )
            {
                const auto one = interfaceConstraints( a, perm, b, lengths, ct );
                constraints.insert( constraints.end(), one.begin(), one.end() );
            }
            const MergeObs o = compareMerge( space, lengths, constraints );
            CHECK( o.id_mismatch == 0 );
            CHECK( o.n_global == o.n_global_expected );
            CHECK( o.sign_mismatch == 0 );

            Agg& g = agg[ct];
            g.n_spaces++;
            g.n_constraints += o.n_constraints;
            g.n_misaligned += o.n_misaligned;
            if( o.n_misaligned > 0 ) g.n_classes_flipping++;
            flips[ct] = o.n_misaligned > 0;
        }
        if( flips.at( ConformingType::Curl ) != flips.at( ConformingType::Divergence ) ) n_disagree++;
    }

    for( const auto& [ct, g] : agg )
        std::cout << "  " << ( ct == ConformingType::Curl ? "H(curl)" : "H(div) " ) << ": spaces "
                  << g.n_spaces << ", constraints " << g.n_constraints << " of which " << g.n_misaligned
                  << " sign-flipping, in " << g.n_classes_flipping << " of 144 classes" << std::endl;
    std::cout << "  classes where the covariant and contravariant rules disagree about flipping: "
              << n_disagree << std::endl;

    // MEASURED REGRESSION INVARIANTS, not derived quantities.  They guard the
    // rules' discriminating power: a rule that degenerated to all-aligned
    // would still satisfy the mismatch checks above on any fixture whose
    // production merge also stopped flipping.  That exactly half of each
    // type's constraints flip, and that H(curl) flips somewhere in 108 of the
    // 144 classes against H(div)'s 72, are observations about this class set;
    // no equidistribution argument is recorded for either.
    CHECK( agg.at( ConformingType::Curl ).n_constraints == 1728 );
    CHECK( agg.at( ConformingType::Curl ).n_misaligned == 864 );
    CHECK( agg.at( ConformingType::Curl ).n_classes_flipping == 108 );
    CHECK( agg.at( ConformingType::Divergence ).n_constraints == 576 );
    CHECK( agg.at( ConformingType::Divergence ).n_misaligned == 288 );
    CHECK( agg.at( ConformingType::Divergence ).n_classes_flipping == 72 );

    // Step 7 demonstrated the two rules' independence on two fixtures: A flips
    // nowhere in H(curl) but twelve times in H(div), D the other way.  Over
    // the full class set they disagree about whether anything flips on half of
    // it, so the permutation parity in the contravariant rule carries
    // information no covariant rule supplies anywhere but at a few points.
    CHECK( n_disagree == 72 );
}


namespace
{
    // STEP 9 (M5 item ii, soft spot b).  find_change returns the FIRST
    // coordinate differing from the frame origin by more than 1e-10 and throws
    // if none does, so coordinateTransform is well defined only when each
    // frame corner differs from the origin in EXACTLY ONE coordinate.  The
    // MARGIN is measured too: a property holding only barely above the
    // tolerance would be fragile rather than true, and find_change is silent
    // rather than diagnostic if two coordinates differ.
    struct FrameObs
    {
        size_t n_frames = 0;
        size_t n_corners = 0;
        size_t n_not_exactly_one = 0; // corners failing the property itself
        size_t n_repeated_axis = 0;   // two corners of one frame changing the same axis
        size_t n_no_partner = 0;      // interface cell whose far side is absent
        double min_changing = std::numeric_limits<double>::max();
        double max_unchanging = 0.0;
    };

    void observeFrame( const SmallVector<param::ParentPoint, 4>& frame, const size_t dim, FrameObs& o )
    {
        o.n_frames++;
        std::set<size_t> axes;
        for( size_t corner = 1; corner <= dim; corner++ )
        {
            o.n_corners++;
            size_t n_changing = 0, which = 0;
            for( size_t i = 0; i < dim; i++ )
            {
                const double a = frame.at( corner ).mPoint( i );
                const double b = frame.at( 0 ).mPoint( i );
                // The same predicate find_change uses, not a restatement of it.
                if( not util::equals( a, b, 1e-10 ) )
                {
                    n_changing++;
                    which = i;
                    o.min_changing = std::min( o.min_changing, std::abs( a - b ) );
                }
                else o.max_unchanging = std::max( o.max_unchanging, std::abs( a - b ) );
            }
            if( n_changing != 1 ) o.n_not_exactly_one++;
            else if( not axes.insert( which ).second ) o.n_repeated_axis++;
        }
    }

    // The two frames coordinateTransform itself builds for one cell: the near
    // one unreversed, the far one across phi_dim with reverse_dart.
    void observeCellFrames( const param::ParametricAtlas& atlas,
                                 const topology::Cell& cell,
                                 const size_t dim,
                                 FrameObs& o )
    {
        observeFrame( param::getFrame( atlas, cell ), dim, o );
        const auto far = phi( atlas.cmap(), dim, cell.dart() );
        if( not far.has_value() )
        {
            o.n_no_partner++;
            return;
        }
        observeFrame( param::getFrame( atlas, topology::Cell( far.value(), dim - 1 ), true ), dim, o );
    }
}

TEST_CASE( "STEP 9: find_change is well defined on every frame in the tested two-patch tensor-product configurations", "[rowd]" )
{
    const KnotVector kv( { 0, 0, 0, 1, 1, 1 }, ptol );
    const auto cmap_1d = std::make_shared<const CombinatorialMap1d>( numElements( kv ) );
    const auto param_1d = std::make_shared<const param::ParametricAtlas1d>( cmap_1d, parametricLengths( kv ) );
    const auto cmap_2d = std::make_shared<const TPCombinatorialMap>( cmap_1d, cmap_1d );
    const auto param_2d = std::make_shared<const param::TPParametricAtlas>( cmap_2d, param_1d, param_1d );
    const auto cmap_3d = std::make_shared<const TPCombinatorialMap>( cmap_2d, cmap_1d );
    const auto param_3d = std::make_shared<const param::TPParametricAtlas>( cmap_3d, param_2d, param_1d );

    // 3d, over the same exhaustion of the 144 interface classes as step 8, and
    // over all four darts of each interface face, because production calls
    // coordinateTransform once per dart.
    FrameObs o3;
    size_t n_built = 0, n_rejected = 0;
    const auto classes = gluingClasses3d( n_built, n_rejected );
    REQUIRE( classes.size() == 144 );
    for( const auto& [cls, conns] : classes )
    {
        const auto mp = buildMultiPatch3d( 2, conns );
        const param::MultiPatchParametricAtlas atlas(
            mp, std::vector<std::shared_ptr<const param::TPParametricAtlas>>{ param_3d, param_3d } );
        for( Dart::IndexType local_d = 0; local_d < 24; local_d++ )
        {
            const Dart gd = mp->toGlobalDart( 0, Dart( local_d ) );
            if( not phi( *mp, 3, gd ).has_value() ) continue;
            observeCellFrames( atlas, Face( gd ), 3, o3 );
        }
    }

    // 2d, swept the same way.  find_change and getFrame are dimension-generic
    // and 2d exercises the three-corner branch, where the third corner is
    // reached by phi^-1 rather than by phi {2,1,1}.
    FrameObs o2;
    size_t n_2d_pairs = 0;
    for( Dart::IndexType da = 0; da < 4; da++ )
    {
        for( Dart::IndexType db = 0; db < 4; db++ )
        {
            std::shared_ptr<const MultiPatchCombinatorialMap> mp;
            try
            {
                mp = std::make_shared<const MultiPatchCombinatorialMap>(
                    std::vector<std::shared_ptr<const TPCombinatorialMap>>{ cmap_2d, cmap_2d },
                    DartConnections{ { { 0, Dart( da ) }, { 1, Dart( db ) } } } );
            }
            catch( const std::exception& )
            {
                continue;
            }
            n_2d_pairs++;
            const param::MultiPatchParametricAtlas atlas(
                mp, std::vector<std::shared_ptr<const param::TPParametricAtlas>>{ param_2d, param_2d } );
            for( Dart::IndexType local_d = 0; local_d < 4; local_d++ )
            {
                const Dart gd = mp->toGlobalDart( 0, Dart( local_d ) );
                if( not phi( *mp, 2, gd ).has_value() ) continue;
                observeCellFrames( atlas, Edge( gd ), 2, o2 );
            }
        }
    }

    // 3d and 2d again on REFINED patches, the case soft spot (b) called "not
    // established in general".  Each frame still lies in one element, whose
    // parent domain is the same unit box, but only a sweep shows that the phi
    // path does not reach a corner belonging to a NEIGHBOURING element, where
    // the parent coordinates would be that element's own.  Interior cells are
    // swept as well as interface ones: getFrame is called at both.
    //
    // The patches are refined UNIFORMLY so that every pair of sides carries
    // the same element decomposition.  Gluing sides whose decompositions
    // differ is not a conforming interface, and the dart-pair constructor
    // accepts it without complaint; see outstanding_issues.txt.
    const KnotVector kv2( { 0, 0, 0, 0.5, 1, 1, 1 }, ptol );
    const auto cmap_1d_2 = std::make_shared<const CombinatorialMap1d>( numElements( kv2 ) );
    const auto param_1d_2 = std::make_shared<const param::ParametricAtlas1d>( cmap_1d_2, parametricLengths( kv2 ) );
    const auto cmap_2d_2 = std::make_shared<const TPCombinatorialMap>( cmap_1d_2, cmap_1d_2 );
    const auto param_2d_2 = std::make_shared<const param::TPParametricAtlas>( cmap_2d_2, param_1d_2, param_1d_2 );
    const auto cmap_3d_2 = std::make_shared<const TPCombinatorialMap>( cmap_2d_2, cmap_1d_2 );
    const auto param_3d_2 = std::make_shared<const param::TPParametricAtlas>( cmap_3d_2, param_2d_2, param_1d_2 );

    // Both the side a dart lies on and its position within that side are
    // functions of d.id() % 24 alone, so sweeping the hex-local positions
    // covers the same 144 classes on a refined patch as on a single element.
    const auto sweepRefined = [&]( const std::shared_ptr<const TPCombinatorialMap>& cmap,
                                   const std::shared_ptr<const param::TPParametricAtlas>& param,
                                   const size_t dim,
                                   const Dart::IndexType n_local,
                                   FrameObs& o_iface,
                                   FrameObs& o_interior,
                                   size_t& n_pairs ) {
        bool interior_done = false;
        for( Dart::IndexType da = 0; da < n_local; da++ )
        {
            for( Dart::IndexType db = 0; db < n_local; db++ )
            {
                std::shared_ptr<const MultiPatchCombinatorialMap> mp;
                try
                {
                    mp = std::make_shared<const MultiPatchCombinatorialMap>(
                        std::vector<std::shared_ptr<const TPCombinatorialMap>>{ cmap, cmap },
                        DartConnections{ { { 0, Dart( da ) }, { 1, Dart( db ) } } } );
                }
                catch( const std::exception& )
                {
                    continue;
                }
                n_pairs++;
                const param::MultiPatchParametricAtlas atlas(
                    mp, std::vector<std::shared_ptr<const param::TPParametricAtlas>>{ param, param } );
                iterateDartsWhile( *mp, [&]( const Dart& gd ) {
                    if( mp->toLocalDart( gd ).first != 0 ) return true;
                    const auto far = phi( *mp, static_cast<int>( dim ), gd );
                    if( not far.has_value() ) return true;
                    // Interface cells for every gluing; interior cells once,
                    // since those do not depend on the gluing.
                    if( mp->toLocalDart( far.value() ).first != 0 )
                        observeCellFrames( atlas, topology::Cell( gd, dim - 1 ), dim, o_iface );
                    else if( not interior_done )
                        observeCellFrames( atlas, topology::Cell( gd, dim - 1 ), dim, o_interior );
                    return true;
                } );
                interior_done = true;
            }
        }
    };

    FrameObs o3r, o3i, o2r, o2i;
    size_t n_3d_refined_pairs = 0, n_2d_refined_pairs = 0;
    sweepRefined( cmap_3d_2, param_3d_2, 3, 24, o3r, o3i, n_3d_refined_pairs );
    sweepRefined( cmap_2d_2, param_2d_2, 2, 4, o2r, o2i, n_2d_refined_pairs );

    std::cout << "STEP 9: find_change well-definedness" << std::endl;
    for( const auto& [label, o] : { std::pair<std::string, FrameObs>{ "3d, one element", o3 },
                                    std::pair<std::string, FrameObs>{ "2d, one element", o2 },
                                    std::pair<std::string, FrameObs>{ "3d refined, interface", o3r },
                                    std::pair<std::string, FrameObs>{ "3d refined, interior", o3i },
                                    std::pair<std::string, FrameObs>{ "2d refined, interface", o2r },
                                    std::pair<std::string, FrameObs>{ "2d refined, interior", o2i } } )
    {
        std::cout << "  " << label << ": frames " << o.n_frames << ", corners " << o.n_corners
                  << ", corners not changing exactly one coordinate " << o.n_not_exactly_one
                  << ", frames repeating an axis " << o.n_repeated_axis << "; smallest change "
                  << o.min_changing << ", largest non-change " << o.max_unchanging << std::endl;
        CHECK( o.n_frames > 0 );
        CHECK( o.n_no_partner == 0 );
        CHECK( o.n_not_exactly_one == 0 );

        // Each corner changes a DIFFERENT axis, so the dim corners span the dim
        // axes and coordinateTransform writes every entry exactly once.  Step 8
        // measured the consequence; this measures the cause.
        CHECK( o.n_repeated_axis == 0 );

        // THE TOLERANCE IS NOT LOAD-BEARING.  On these parent domains the
        // separation is total: coordinates that change do so by a full parent
        // extent and coordinates that do not are bitwise equal, so the 1e-10
        // threshold sits many orders of magnitude inside the gap rather than
        // adjudicating near it.
        CHECK( o.max_unchanging == 0.0 );
        CHECK( o.min_changing > 1e-6 );
    }
    CHECK( n_2d_pairs == 16 );
    CHECK( n_3d_refined_pairs == 576 );
    CHECK( n_2d_refined_pairs == 16 );
    std::cout << "  dart pairs built: 2d one element " << n_2d_pairs << " of 16, 3d refined "
              << n_3d_refined_pairs << " of 576, 2d refined " << n_2d_refined_pairs << " of 16"
              << std::endl;
}
