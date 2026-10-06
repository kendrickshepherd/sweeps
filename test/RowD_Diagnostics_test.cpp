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
