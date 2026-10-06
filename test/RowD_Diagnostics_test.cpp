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
#include <iostream>
#include <set>
#include <vector>

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
