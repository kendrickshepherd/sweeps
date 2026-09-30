// Stage 1 evidence, converted from scratchpad probes into executable gates.
// See outstanding_issues.txt, STAGE 1 RESULTS.
//
// Every test here is PERMANENT.  Three of them were written as
// [known-defect][!shouldfail] requirements; Stage 2's side-aware consumers
// removed the TAG, never the case:
//
//   invariants   - phi and both function-ID merges are CORRECT, so what phi
//                  produces is pinned directly.  These must keep passing,
//                  INCLUDING after the shared transform is relocated
//                  (possibly into the topology layer).
//
//   requirements - that the trace and validator consumers agree with phi.
//                  While the consumers were side-blind these carried
//                  [!shouldfail] so Catch2 inverted the verdict.  They guard
//                  the transform at a lower layer than the spline fixtures.
//
// Each of those three cases contains exactly ONE aggregate assertion, so a
// partial fix cannot hide behind another still-failing row - the single
// failure message enumerates every case that remains wrong.  That is why the
// loops in those cases record problems into a list instead of asserting.
//
// The expected transforms are restated here INDEPENDENTLY of production code,
// as a specification, rather than read back from it: a test that re-derives
// the rule detects a change in phi, whereas one that calls the same helper
// cannot.  No assertion below pins a value that Stage 1 identified as wrong.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <CombinatorialMap1d.hpp>
#include <CombinatorialMapMethods.hpp>
#include <KnotVector.hpp>
#include <MultiPatchCombinatorialMap.hpp>
#include <MultiPatchSplineFactory.hpp>
#include <ParametricAtlas.hpp>
#include <SideCoordinateTransform.hpp>
#include <TPCombinatorialMap.hpp>
#include <TPParametricAtlas.hpp>
#include <TraceMesh.hpp>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace basis;
using namespace topology;

namespace
{
    using InternalConnectionsMap = MultiPatchCombinatorialMap::InternalConnectionsMap;
    using ConstituentSide = MultiPatchCombinatorialMap::ConstituentSide;
    using TPPermutation = MultiPatchCombinatorialMap::TPPermutation;

    constexpr double ptol = 1e-10;
    constexpr size_t n_elem = 3; // elements per axis in every fixture here

    // 2d side ids: 0 = s-upper, 1 = s-lower, 2 = t-upper, 3 = t-lower.
    KnotVector uniformKV() { return KnotVector( { 0, 0, 0, 1, 2, 3, 3, 3 }, ptol ); }

    std::shared_ptr<const TPSplineSpace> makePatch( const SmallVector<KnotVector, 3>& kvs,
                                                    const SmallVector<size_t, 3>& degrees )
    {
        return std::make_shared<const TPSplineSpace>( buildBSpline( kvs, degrees ) );
    }

    std::shared_ptr<const TPSplineSpace> squarePatch()
    {
        const KnotVector kv = uniformKV();
        return makePatch( { kv, kv }, { 2, 2 } );
    }

    InternalConnectionsMap conn2d( const size_t sa, const size_t sb, const TPPermutation p )
    {
        const ConstituentSide a{ 0, sa };
        const ConstituentSide b{ 1, sb };
        return { { a, { p, b } }, { b, { p, a } } };
    }

    // ---------------- the specification, restated independently ---------------------

    // F_s in 2d: the induced boundary traversal opposes increasing parameter on
    // sides 1 and 2 (MultiPatchCombinatorialMap::flipNecessaryDarts, 2d branch).
    bool sideFlips2d( const size_t s ) { return s == 1 or s == 2; }

    // F_s in 3d flips tangential axis 0 on sides 0, 3 and 4 - a DIFFERENT set.
    bool sideFlips3d( const size_t s ) { return s == 0 or s == 3 or s == 4; }

    size_t flipIdx( const size_t i, const size_t n ) { return n - 1 - i; }

    // phi's total map is F_b o P o F_a.  In 2d, P is a constant (a 1d side has
    // only one permutation), so the parity of the two side flips decides
    // alignment: reversed iff the flip count is EVEN.
    size_t expected2d( const size_t i, const size_t sa, const size_t sb, const size_t n )
    {
        const size_t flips = ( sideFlips2d( sa ) ? 1u : 0u ) + ( sideFlips2d( sb ) ? 1u : 0u );
        return ( flips % 2 == 0 ) ? flipIdx( i, n ) : i;
    }

    std::vector<size_t> applyPerm3d( const std::vector<size_t>& v, const TPPermutation p, const size_t n )
    {
        switch( p )
        {
            case TPPermutation::ZeroToZero:  return { flipIdx( v.at( 0 ), n ), v.at( 1 ) };
            case TPPermutation::ZeroToOne:   return { flipIdx( v.at( 1 ), n ), flipIdx( v.at( 0 ), n ) };
            case TPPermutation::ZeroToTwo:   return { v.at( 0 ), flipIdx( v.at( 1 ), n ) };
            case TPPermutation::ZeroToThree: return { v.at( 1 ), v.at( 0 ) };
            case TPPermutation::Flip1d:      return { flipIdx( v.at( 0 ), n ) };
        }
        return v;
    }

    std::vector<size_t> applyFlip3d( std::vector<size_t> v, const size_t s, const size_t n )
    {
        if( sideFlips3d( s ) ) v.at( 0 ) = flipIdx( v.at( 0 ), n );
        return v;
    }

    std::vector<size_t> expected3d( const std::vector<size_t>& v, const size_t sa, const size_t sb,
                                   const TPPermutation p, const size_t n )
    {
        return applyFlip3d( applyPerm3d( applyFlip3d( v, sa, n ), p, n ), sb, n );
    }

    // ---------------- measurement helpers -------------------------------------------

    // tangential element index of a 2d dart's element, relative to a side.
    size_t tangIdx2d( const TPCombinatorialMap& cmap, const Dart& d, const size_t side_id )
    {
        return unflattenFull( cmap, d ).unflat_darts.at( 1 - side_id / 2 ).id(); // normal axis is side_id/2
    }

    std::vector<size_t> tang3d( const SmallVector<Dart, 3>& unflat, const size_t side_id )
    {
        std::vector<size_t> out;
        for( size_t a = 0; a < unflat.size(); a++ )
            if( a != side_id / 2 ) out.push_back( unflat.at( a ).id() );
        return out;
    }

    std::optional<size_t> faceSideId( const param::ParametricAtlas& atlas, const Dart& d )
    {
        const param::BaryCoordIsZeroVec b = param::parentDomainBoundary( atlas, Face( d ) );
        std::optional<size_t> found;
        for( size_t i = 0; i < b.size(); i++ )
            if( b.at( i ) )
            {
                if( found.has_value() ) return std::nullopt; // an edge, not a face interior
                found = i;
            }
        return found;
    }

    // phi's element correspondence across a 2d interface, from_patch -> the other.
    // `consistent` reports whether every dart of an element agreed on the pairing.
    std::map<size_t, size_t> phiCorr2d( const MultiPatchCombinatorialMap& mp,
                                        const size_t from_patch,
                                        const size_t from_side,
                                        const size_t to_side,
                                        bool& consistent )
    {
        std::map<size_t, size_t> out;
        consistent = true;
        const auto& src = *mp.constituents().at( from_patch );
        iterateDartsWhile( src, [&]( const Dart& d ) {
            const auto n = topology::phi( mp, 2, mp.toGlobalDart( from_patch, d ) );
            if( not n.has_value() ) return true;
            const auto [q, nd] = mp.toLocalDart( *n );
            if( q == from_patch ) return true;
            const size_t key = tangIdx2d( src, d, from_side );
            const size_t val = tangIdx2d( *mp.constituents().at( q ), nd, to_side );
            const auto it = out.find( key );
            if( it != out.end() and it->second != val ) consistent = false;
            out[key] = val;
            return true;
        } );
        return out;
    }

    // the same correspondence as the trace layer computes it, patch 0 -> patch 1.
    std::map<size_t, size_t> traceCorr2d( const SplineSpace& ss,
                                          const MultiPatchCombinatorialMap& mp,
                                          const size_t sa,
                                          const size_t sb )
    {
        std::map<size_t, size_t> out;
        for( const TraceMeshInterface& iface : patchTraceMeshInterfaces( ss ) )
        {
            if( not iface.second.has_value() ) continue;
            const auto [pa, da] = mp.toLocalDart( iface.first.element.dart() );
            const auto [pb, db] = mp.toLocalDart( iface.second->element.dart() );
            if( pa == 0 and pb == 1 )
                out[tangIdx2d( *mp.constituents().at( 0 ), da, sa )] =
                    tangIdx2d( *mp.constituents().at( 1 ), db, sb );
            else if( pa == 1 and pb == 0 )
                out[tangIdx2d( *mp.constituents().at( 0 ), db, sa )] =
                    tangIdx2d( *mp.constituents().at( 1 ), da, sb );
        }
        return out;
    }

    std::string describe( const std::map<size_t, size_t>& m )
    {
        std::ostringstream os;
        for( const auto& [k, v] : m ) os << " " << k << "->" << v;
        return os.str();
    }

    std::string describeVec( const std::vector<size_t>& v )
    {
        std::ostringstream os;
        os << "(";
        for( size_t i = 0; i < v.size(); i++ ) os << ( i ? "," : "" ) << v.at( i );
        os << ")";
        return os.str();
    }

    std::string joined( const std::vector<std::string>& lines )
    {
        std::ostringstream os;
        os << lines.size() << " case(s):";
        for( const std::string& l : lines ) os << "\n  " << l;
        return os.str();
    }

    // B3/B4: one axis mirrored against itself, sides 0 <-> 1.
    MultiPatchSplineSpace buildMirroredPair( const bool mirrored )
    {
        const KnotVector kv_s( { 0, 0, 0, 1, 1, 1 }, ptol );
        const KnotVector kv_a( { 0, 0, 0, 1, 3, 3, 3 }, ptol ); // spans [1,2]
        const KnotVector kv_b( { 0, 0, 0, 2, 3, 3, 3 }, ptol ); // spans [2,1]
        return buildH1MultiPatchSplineSpace(
            { makePatch( { kv_s, kv_a }, { 2, 2 } ),
              makePatch( { kv_s, mirrored ? kv_b : kv_a }, { 2, 2 } ) },
            conn2d( 0, 1, TPPermutation::Flip1d ) );
    }

    // Side pairs whose F_b o P o F_a equals P, so the values the cases below
    // were pinned against are unchanged by the side-aware signature.
    SideCoordinateTransform permP3d( const TPPermutation p )
    {
        return sideCoordinateTransform( 3, 1, 2, p );
    }

    SideCoordinateTransform permP2d()
    {
        return sideCoordinateTransform( 2, 0, 3, TPPermutation::Flip1d );
    }

}

// ===================== map semantics: these hold today =========================

TEST_CASE( "phi is an orientation-consistent involution at every 2d patch interface" )
{
    const auto patch = squarePatch();
    for( size_t sa = 0; sa < 4; sa++ )
    {
        for( size_t sb = 0; sb < 4; sb++ )
        {
            CAPTURE( sa, sb );
            const MultiPatchSplineSpace h1 =
                buildH1MultiPatchSplineSpace( { patch, patch }, conn2d( sa, sb, TPPermutation::Flip1d ) );
            const auto& mp = h1.basisComplex().parametricAtlas().cmap();

            size_t n_iface = 0;
            iterateDartsWhile( *mp.constituents().at( 0 ), [&]( const Dart& d ) {
                const Dart gd = mp.toGlobalDart( 0, d );
                const auto n = topology::phi( mp, 2, gd );
                if( not n.has_value() ) return true;
                if( mp.toLocalDart( *n ).first == 0 ) return true;
                n_iface++;
                const auto back = topology::phi( mp, 2, *n );
                REQUIRE( back.has_value() );
                CHECK( *back == gd );                        // involution
                CHECK_FALSE( onSameVertex( mp, gd, *n ) );   // phi_i moves the originating vertex
                const auto nxt = topology::phi( mp, 1, gd );
                REQUIRE( nxt.has_value() );
                CHECK( onSameVertex( mp, *nxt, *n ) );       // same edge, opposite traversal
                return true;
            } );
            CHECK( n_iface == n_elem );
        }
    }
}

TEST_CASE( "the interface orientation convention matches the patch interior" )
{
    // The interior path is TPCombinatorialMap, which is not under suspicion, so
    // agreement here shows the convention is the repository's own rather than an
    // artifact of crossing a patch boundary.
    const auto patch = squarePatch();
    for( const auto [sa, sb] : { std::pair<size_t, size_t>{ 0, 1 }, std::pair<size_t, size_t>{ 0, 3 } } )
    {
        CAPTURE( sa, sb );
        const MultiPatchSplineSpace h1 =
            buildH1MultiPatchSplineSpace( { patch, patch }, conn2d( sa, sb, TPPermutation::Flip1d ) );
        const auto& mp = h1.basisComplex().parametricAtlas().cmap();

        size_t n_interior = 0, n_interface = 0;
        iterateDartsWhile( *mp.constituents().at( 0 ), [&]( const Dart& d ) {
            const Dart gd = mp.toGlobalDart( 0, d );
            const auto n = topology::phi( mp, 2, gd );
            if( not n.has_value() ) return true;
            ( mp.toLocalDart( *n ).first != 0 ) ? n_interface++ : n_interior++;
            CHECK_FALSE( onSameVertex( mp, gd, *n ) ); // uniform, interior and interface alike
            return true;
        } );
        CHECK( n_interior > 0 );
        CHECK( n_interface == n_elem );
    }
}

TEST_CASE( "phi's element correspondence for all 16 ordered 2d side pairs" )
{
    // phi is authoritative, so its correspondence is pinned against the
    // independently restated rule F_b o P o F_a.
    const auto patch = squarePatch();
    for( size_t sa = 0; sa < 4; sa++ )
    {
        for( size_t sb = 0; sb < 4; sb++ )
        {
            CAPTURE( sa, sb );
            const MultiPatchSplineSpace h1 =
                buildH1MultiPatchSplineSpace( { patch, patch }, conn2d( sa, sb, TPPermutation::Flip1d ) );
            const auto& mp = h1.basisComplex().parametricAtlas().cmap();

            bool consistent = false;
            const std::map<size_t, size_t> corr = phiCorr2d( mp, 0, sa, sb, consistent );
            CHECK( consistent ); // every dart of an element agrees on the pairing
            REQUIRE( corr.size() == n_elem );
            for( size_t i = 0; i < n_elem; i++ )
            {
                CAPTURE( i );
                CHECK( corr.at( i ) == expected2d( i, sa, sb, n_elem ) );
            }
        }
    }
}

TEST_CASE( "phi's correspondence is the same map read from either side" )
{
    // Source/destination order reversal: the two stored directions of an
    // interface must be inverse correspondences, not merely both present.
    const auto patch = squarePatch();
    for( size_t sa = 0; sa < 4; sa++ )
    {
        for( size_t sb = 0; sb < 4; sb++ )
        {
            CAPTURE( sa, sb );
            const MultiPatchSplineSpace h1 =
                buildH1MultiPatchSplineSpace( { patch, patch }, conn2d( sa, sb, TPPermutation::Flip1d ) );
            const auto& mp = h1.basisComplex().parametricAtlas().cmap();

            bool c_fwd = false, c_rev = false;
            const std::map<size_t, size_t> fwd = phiCorr2d( mp, 0, sa, sb, c_fwd );
            const std::map<size_t, size_t> rev = phiCorr2d( mp, 1, sb, sa, c_rev );
            CHECK( c_fwd );
            CHECK( c_rev );
            REQUIRE( fwd.size() == n_elem );
            REQUIRE( rev.size() == n_elem );
            for( const auto& [i, j] : fwd )
            {
                CAPTURE( i, j );
                REQUIRE( rev.count( j ) == 1 );
                CHECK( rev.at( j ) == i );
            }
        }
    }
}

TEST_CASE( "trace side index transform is a bijection preserving bounds" )
{
    // A swapping permutation is only meaningful when the tangential extents
    // match - it sends one axis' indices into the other axis - so rectangular
    // interfaces are checked against the non-swapping permutations only.
    const std::vector<TPPermutation> non_swapping{ TPPermutation::ZeroToZero, TPPermutation::ZeroToTwo };
    const std::vector<TPPermutation> all_2d{ TPPermutation::ZeroToZero,
                                             TPPermutation::ZeroToOne,
                                             TPPermutation::ZeroToTwo,
                                             TPPermutation::ZeroToThree };

    for( const auto& lengths : { std::vector<size_t>{ 4, 5 }, std::vector<size_t>{ 3, 3 } } )
    {
        const bool square = lengths.at( 0 ) == lengths.at( 1 );
        for( const TPPermutation p : ( square ? all_2d : non_swapping ) )
        {
            CAPTURE( lengths.at( 0 ), lengths.at( 1 ), static_cast<int>( p ) );
            std::set<std::vector<size_t>> images;
            for( size_t i = 0; i < lengths.at( 0 ); i++ )
            {
                for( size_t j = 0; j < lengths.at( 1 ); j++ )
                {
                    const std::vector<size_t> out = permuteTraceSideIndex( permP3d( p ), { i, j }, lengths );
                    REQUIRE( out.size() == 2 );
                    CHECK( out.at( 0 ) < lengths.at( 0 ) );
                    CHECK( out.at( 1 ) < lengths.at( 1 ) );
                    images.insert( out );
                }
            }
            CHECK( images.size() == lengths.at( 0 ) * lengths.at( 1 ) ); // bijection
        }
    }

    for( const size_t n : { 1u, 2u, 3u, 7u } )
    {
        CAPTURE( n );
        std::set<std::vector<size_t>> images;
        for( size_t i = 0; i < n; i++ )
        {
            const std::vector<size_t> out = permuteTraceSideIndex( permP2d(), { i }, { n } );
            REQUIRE( out.size() == 1 );
            CHECK( out.at( 0 ) < n );
            images.insert( out );
        }
        CHECK( images.size() == n );
    }
}

TEST_CASE( "axis-swapping permutations draw each output axis from the other input axis" )
{
    // Where an axis swap is actually observable: a NON-square grid.  Asserted
    // as provenance - which input an output depends on - rather than as values,
    // because the single `lengths` argument measures a flipped swapped axis
    // against the wrong extent.  Stage 2's source/destination extents should
    // remove that limitation without changing the provenance asserted here.
    const std::vector<size_t> lengths{ 4, 5 };
    for( const TPPermutation p : { TPPermutation::ZeroToOne, TPPermutation::ZeroToThree } )
    {
        CAPTURE( static_cast<int>( p ) );
        std::set<size_t> out0, out1;
        for( size_t i = 0; i < lengths.at( 0 ); i++ )
        {
            const std::vector<size_t> out = permuteTraceSideIndex( permP3d( p ), { i, 2 }, lengths );
            REQUIRE( out.size() == 2 );
            out0.insert( out.at( 0 ) );
            out1.insert( out.at( 1 ) );
        }
        CHECK( out0.size() == 1 );                 // output axis 0 ignores input axis 0
        CHECK( out1.size() == lengths.at( 0 ) );   // output axis 1 carries input axis 0
    }

    // The pure swap carries values across untouched, whatever the extents are.
    CHECK( permuteTraceSideIndex( permP3d( TPPermutation::ZeroToThree ), { 1, 4 }, { 4, 5 } ) == std::vector<size_t>{ 4, 1 } );
    CHECK( permuteTraceSideIndex( permP3d( TPPermutation::ZeroToThree ), { 1, 3 }, { 5, 4 } ) == std::vector<size_t>{ 3, 1 } );
}

TEST_CASE( "trace side transforms are involutions on square interfaces" )
{
    const std::vector<size_t> lengths{ 3, 3 };
    for( const TPPermutation p : { TPPermutation::ZeroToZero,
                                   TPPermutation::ZeroToOne,
                                   TPPermutation::ZeroToTwo,
                                   TPPermutation::ZeroToThree } )
    {
        CAPTURE( static_cast<int>( p ) );
        for( size_t i = 0; i < lengths.at( 0 ); i++ )
        {
            for( size_t j = 0; j < lengths.at( 1 ); j++ )
            {
                const std::vector<size_t> once = permuteTraceSideIndex( permP3d( p ), { i, j }, lengths );
                CHECK( permuteTraceSideIndex( permP3d( p ), once, lengths ) == std::vector<size_t>{ i, j } );
            }
        }
        const Eigen::Vector2d pt( 0.2, 0.7 );
        CHECK( ( permuteTraceSidePoint( permP3d( p ), permuteTraceSidePoint( permP3d( p ), pt ) ) - pt ).norm() < 1e-14 );
    }

    for( size_t i = 0; i < n_elem; i++ )
    {
        const std::vector<size_t> once = permuteTraceSideIndex( permP2d(), { i }, { n_elem } );
        CHECK( permuteTraceSideIndex( permP2d(), once, { n_elem } ) == std::vector<size_t>{ i } );
    }
}

TEST_CASE( "trace side index and point transforms describe the same map" )
{
    // Element i spans [i/n, (i+1)/n], so its centre is (i+0.5)/n.  Whatever the
    // transform does to indices it must do to points.  Stage 2 should make this
    // structural by deriving both from one transform; until then, asserted.
    const size_t n = n_elem;
    const std::vector<size_t> lengths{ n, n };
    const auto centreOf = []( const size_t i ) { return ( static_cast<double>( i ) + 0.5 ) / static_cast<double>( n ); };

    for( const TPPermutation p : { TPPermutation::ZeroToZero,
                                   TPPermutation::ZeroToOne,
                                   TPPermutation::ZeroToTwo,
                                   TPPermutation::ZeroToThree } )
    {
        for( size_t i = 0; i < n; i++ )
        {
            for( size_t j = 0; j < n; j++ )
            {
                CAPTURE( static_cast<int>( p ), i, j );
                const std::vector<size_t> idx = permuteTraceSideIndex( permP3d( p ), { i, j }, lengths );
                const Eigen::VectorXd moved =
                    permuteTraceSidePoint( permP3d( p ), Eigen::Vector2d( centreOf( i ), centreOf( j ) ) );
                REQUIRE( moved.size() == 2 );
                CHECK_THAT( moved( 0 ), Catch::Matchers::WithinAbs( centreOf( idx.at( 0 ) ), 1e-13 ) );
                CHECK_THAT( moved( 1 ), Catch::Matchers::WithinAbs( centreOf( idx.at( 1 ) ), 1e-13 ) );
            }
        }
    }

    for( size_t i = 0; i < n; i++ )
    {
        CAPTURE( i );
        const std::vector<size_t> idx = permuteTraceSideIndex( permP2d(), { i }, { n } );
        CHECK_THAT( permuteTraceSidePoint( permP2d(), Eigen::Vector<double, 1>( centreOf( i ) ) )( 0 ),
                    Catch::Matchers::WithinAbs( centreOf( idx.at( 0 ) ), 1e-13 ) );
    }
}

TEST_CASE( "coordinateTransform derives component maps from phi-connected frames" )
{
    // coordinateTransform never reads the stored permutation - it compares the
    // actual parent-point frames on both sides - so it is correct by
    // construction and is NOT part of the side-blind defect.
    const KnotVector kv_s( { 0, 0, 0, 1, 1, 1 }, ptol );
    const KnotVector kv_t( { 0, 0, 0, 1, 1, 2, 2, 2 }, ptol );
    const auto p_st = makePatch( { kv_s, kv_t }, { 2, 2 } );
    const auto p_ts = makePatch( { kv_t, kv_s }, { 2, 2 } );

    // Rotated gluing: patch 1's axes are transposed, so the component map must
    // be a genuine swap rather than the identity.
    const MultiPatchSplineSpace h1 =
        buildH1MultiPatchSplineSpace( { p_st, p_ts }, conn2d( 0, 3, TPPermutation::Flip1d ) );
    const auto& atlas = h1.basisComplex().parametricAtlas();
    const auto& mp = atlas.cmap();

    size_t n_checked = 0;
    for( size_t p = 0; p < 2; p++ )
    {
        const param::TPParametricAtlas& sub = h1.subSpaces().at( p )->basisComplex().parametricAtlas();
        for( const topology::Cell& cc : param::cornerCells( sub, mp.dim() - 1 ) )
        {
            const topology::Cell g( mp.toGlobalDart( p, cc.dart() ), cc.dim() );
            const auto nb = topology::phi( mp, mp.dim(), g.dart() );
            if( not nb.has_value() ) continue;
            if( mp.toLocalDart( *nb ).first == p ) continue;
            CAPTURE( p, g.dart().id() );
            const auto transform = param::coordinateTransform( atlas, g );
            REQUIRE( transform.size() == mp.dim() );
            std::set<size_t> targets;
            for( const auto& [component, aligned] : transform ) targets.insert( component );
            CHECK( targets.size() == mp.dim() );     // a permutation of components
            CHECK( transform.at( 0 ).first == 1 );   // transposed patch -> genuine swap
            CHECK( transform.at( 1 ).first == 0 );
            n_checked++;
        }
    }
    CHECK( n_checked > 0 );
}

TEST_CASE( "regression tripwire: interface function counts cannot distinguish correspondences" )
{
    // A tripwire, NOT evidence of correctness.  The point is not the value but
    // that it is IDENTICAL for all 16 side pairs, including the eight where phi
    // and the trace layer disagree - one of the three reasons the suite stayed
    // green.  Only per-paired-element shared sets can discriminate.
    const auto patch = squarePatch();
    std::set<size_t> counts;
    for( size_t sa = 0; sa < 4; sa++ )
        for( size_t sb = 0; sb < 4; sb++ )
            counts.insert( buildH1MultiPatchSplineSpace(
                               { patch, patch }, conn2d( sa, sb, TPPermutation::Flip1d ) )
                               .numFunctions() );
    CHECK( counts.size() == 1 );                    // the invariance is the finding
    CHECK( *counts.begin() == 45 );                 // raw 25 + 25, five interface functions merged
}

TEST_CASE( "connectionsOfSweptMultipatch preserves the 2d in-plane correspondence" )
{
    // A sweep PRODUCES permutation data rather than consuming it, so its
    // semantics must be settled before the shared transform's contract is
    // frozen.  Requirement: the in-plane correspondence is exactly what the 2d
    // map had, and the swept axis maps layer to layer.
    //
    // ZeroToZero flips tangential axis 0 and keeps axis 1.  For every lateral
    // side, tangential axis 1 IS the swept axis, so that is the only one of the
    // four permutations that leaves the sweep aligned while still reversing
    // in-plane.  It agrees with 2d because g2 = {0,1,1,0} and g3 = {1,0,0,1}
    // are complements on the lateral sides, and the parity rule adds two of
    // them, so the two complements cancel.
    const KnotVector kv = uniformKV();
    const KnotVector kv_sweep( { 0, 0, 0, 1, 2, 2, 2 }, ptol ); // 2 layers
    const auto patch2d = makePatch( { kv, kv }, { 2, 2 } );
    const auto patch3d = makePatch( { kv, kv, kv_sweep }, { 2, 2, 2 } );

    for( size_t sa = 0; sa < 4; sa++ )
    {
        for( size_t sb = 0; sb < 4; sb++ )
        {
            CAPTURE( sa, sb );
            const InternalConnectionsMap conns_2d = conn2d( sa, sb, TPPermutation::Flip1d );
            const InternalConnectionsMap conns_3d = topology::connectionsOfSweptMultipatch( conns_2d );

            REQUIRE( conns_3d.size() == conns_2d.size() );
            for( const auto& [side, perm_and_other] : conns_3d )
            {
                CHECK( perm_and_other.first == TPPermutation::ZeroToZero );
                REQUIRE( conns_2d.count( side ) == 1 );
                CHECK( conns_2d.at( side ).second.side_id == perm_and_other.second.side_id );
            }

            const MultiPatchSplineSpace h1_2d = buildH1MultiPatchSplineSpace( { patch2d, patch2d }, conns_2d );
            bool consistent = false;
            const std::map<size_t, size_t> corr_2d =
                phiCorr2d( h1_2d.basisComplex().parametricAtlas().cmap(), 0, sa, sb, consistent );
            CHECK( consistent );
            REQUIRE( corr_2d.size() == n_elem );

            const MultiPatchSplineSpace h1_3d = buildH1MultiPatchSplineSpace( { patch3d, patch3d }, conns_3d );
            const auto& mp3 = h1_3d.basisComplex().parametricAtlas().cmap();
            const auto& src = *mp3.constituents().at( 0 );
            size_t n_seen = 0;
            iterateDartsWhile( src, [&]( const Dart& d ) {
                const auto n = topology::phi( mp3, 3, mp3.toGlobalDart( 0, d ) );
                if( not n.has_value() ) return true;
                const auto [q, nd] = mp3.toLocalDart( *n );
                if( q == 0 ) return true;
                const std::vector<size_t> ta = tang3d( unflattenFull( src, d ).unflat_darts, sa );
                const std::vector<size_t> tb =
                    tang3d( unflattenFull( *mp3.constituents().at( q ), nd ).unflat_darts, sb );
                REQUIRE( ta.size() == 2 );
                REQUIRE( tb.size() == 2 );
                CAPTURE( ta.at( 0 ), ta.at( 1 ), tb.at( 0 ), tb.at( 1 ) );
                CHECK( tb.at( 1 ) == ta.at( 1 ) );               // swept axis: layer to layer
                REQUIRE( corr_2d.count( ta.at( 0 ) ) == 1 );
                CHECK( tb.at( 0 ) == corr_2d.at( ta.at( 0 ) ) ); // in-plane: exactly the 2d map
                n_seen++;
                return true;
            } );
            CHECK( n_seen > 0 );
        }
    }
}

TEST_CASE( "phi's 3d interface map is F_b o P o F_a for every side pair and permutation" )
{
    // Exhaustive over all 6 x 6 x 4 = 144 (source side, destination side,
    // permutation) triples.  Pure topology: no knot vectors, no spline spaces
    // and no validator, which is what makes exhaustive coverage cheap and also
    // what lets every triple be reached at all - the spline builder rejects
    // some of them on knot compatibility before phi is ever consulted.
    //
    // The expected value is restated from the rule, not read back from phi, so
    // this detects a change in phi rather than agreeing with it by construction.
    const auto line = std::make_shared<const CombinatorialMap1d>( n_elem );
    const auto sq = std::make_shared<const TPCombinatorialMap>( line, line );
    const auto cube = std::make_shared<const TPCombinatorialMap>( sq, line );

    size_t n_triples = 0, n_pairs = 0;
    std::vector<std::string> mismatches;
    for( size_t sa = 0; sa < 6; sa++ )
    for( size_t sb = 0; sb < 6; sb++ )
    for( const TPPermutation p : { TPPermutation::ZeroToZero, TPPermutation::ZeroToOne,
                                   TPPermutation::ZeroToTwo, TPPermutation::ZeroToThree } )
    {
        n_triples++;
        const ConstituentSide a{ 0, sa };
        const ConstituentSide b{ 1, sb };
        const MultiPatchCombinatorialMap mp( { cube, cube },
                                             InternalConnectionsMap{ { a, { p, b } }, { b, { p, a } } } );

        std::map<std::vector<size_t>, std::vector<size_t>> from_phi;
        iterateDartsWhile( *cube, [&]( const Dart& d ) {
            const auto n = topology::phi( mp, 3, mp.toGlobalDart( 0, d ) );
            if( not n.has_value() ) return true;
            const auto [q, nd] = mp.toLocalDart( *n );
            if( q != 1 ) return true;
            from_phi[tang3d( unflattenFull( *cube, d ).unflat_darts, sa )] =
                tang3d( unflattenFull( *cube, nd ).unflat_darts, sb );
            return true;
        } );

        std::ostringstream os;
        os << "sides " << sa << " <-> " << sb << " perm " << static_cast<int>( p );
        if( from_phi.size() != n_elem * n_elem )
        {
            mismatches.push_back( os.str() + ": phi paired " + std::to_string( from_phi.size() ) +
                                  " interface elements, expected " + std::to_string( n_elem * n_elem ) );
            continue;
        }
        std::set<std::vector<size_t>> images;
        for( const auto& [ta, tb] : from_phi )
        {
            n_pairs++;
            images.insert( tb );
            const std::vector<size_t> spec = expected3d( ta, sa, sb, p, n_elem );
            if( tb != spec )
                mismatches.push_back( os.str() + ": at " + describeVec( ta ) + " phi " + describeVec( tb ) +
                                      " but F_b o P o F_a gives " + describeVec( spec ) );
        }
        if( images.size() != from_phi.size() )
            mismatches.push_back( os.str() + ": phi is not injective on the interface" );
    }

    INFO( "phi departs from the restated rule on " << joined( mismatches ) );
    CHECK( mismatches.empty() );
    CHECK( n_triples == 144 );
    CHECK( n_pairs == 144 * n_elem * n_elem );
}

// ---------------- the shared transform, Stage 2 step 3 ---------------------------

TEST_CASE( "sideCoordinateTransform reproduces the restated interface rule exhaustively" )
{
    // Compared against the SAME independently restated spec the phi cases use,
    // never against phi's output.  The phi case above pins phi to that spec on
    // all 144 triples, so agreement with phi follows transitively and phi need
    // not be re-extracted here.
    const size_t n = n_elem;
    std::vector<std::string> wrong;
    size_t n_2d = 0, n_3d = 0;

    for( size_t sa = 0; sa < 4; sa++ )
    for( size_t sb = 0; sb < 4; sb++ )
    {
        n_2d++;
        const SideCoordinateTransform t = sideCoordinateTransform( 2, sa, sb, TPPermutation::Flip1d );
        for( size_t i = 0; i < n; i++ )
        {
            const std::vector<size_t> got = transformSideIndex( t, { i }, { n } );
            const std::vector<size_t> spec{ expected2d( i, sa, sb, n ) };
            if( got != spec )
                wrong.push_back( "2d sides " + std::to_string( sa ) + " <-> " + std::to_string( sb ) + " at " +
                                 std::to_string( i ) + ": transform " + describeVec( got ) + " spec " +
                                 describeVec( spec ) );
        }
    }

    for( size_t sa = 0; sa < 6; sa++ )
    for( size_t sb = 0; sb < 6; sb++ )
    for( const TPPermutation p : { TPPermutation::ZeroToZero, TPPermutation::ZeroToOne,
                                   TPPermutation::ZeroToTwo, TPPermutation::ZeroToThree } )
    {
        n_3d++;
        const SideCoordinateTransform t = sideCoordinateTransform( 3, sa, sb, p );
        std::ostringstream os;
        os << "3d sides " << sa << " <-> " << sb << " perm " << static_cast<int>( p );
        for( size_t i = 0; i < n; i++ )
        for( size_t j = 0; j < n; j++ )
        {
            const std::vector<size_t> got = transformSideIndex( t, { i, j }, { n, n } );
            const std::vector<size_t> spec = expected3d( { i, j }, sa, sb, p, n );
            if( got != spec )
                wrong.push_back( os.str() + " at " + describeVec( { i, j } ) + ": transform " +
                                 describeVec( got ) + " spec " + describeVec( spec ) );
        }
    }

    INFO( "transform departs from the restated rule on " << joined( wrong ) );
    CHECK( wrong.empty() );
    CHECK( n_2d == 16 );
    CHECK( n_3d == 144 );
}

TEST_CASE( "the shared transform reports the destination shape on non-square sides" )
{
    // What a source-lengths-only signature cannot express.  The old arithmetic
    // is sound on non-square sides - each flip already uses its own axis's
    // extent - but the caller has nothing to compare the destination lattice
    // against, so a non-conforming interface cannot be detected.  Every flip
    // touching the value from source axis s uses n_s, which is also the extent
    // of the destination axis it lands on, so the parity composition holds
    // when the two extents differ.
    const std::vector<size_t> src{ 4, 5 };
    std::vector<std::string> wrong;

    for( size_t sa = 0; sa < 6; sa++ )
    for( size_t sb = 0; sb < 6; sb++ )
    for( const TPPermutation p : { TPPermutation::ZeroToZero, TPPermutation::ZeroToOne,
                                   TPPermutation::ZeroToTwo, TPPermutation::ZeroToThree } )
    {
        const SideCoordinateTransform t = sideCoordinateTransform( 3, sa, sb, p );
        const std::vector<size_t> dst = transformedExtents( t, src );
        std::ostringstream os;
        os << "sides " << sa << " <-> " << sb << " perm " << static_cast<int>( p );

        const bool swaps = p == TPPermutation::ZeroToOne or p == TPPermutation::ZeroToThree;
        const std::vector<size_t> want =
            swaps ? std::vector<size_t>{ src.at( 1 ), src.at( 0 ) } : src;
        if( dst != want )
            wrong.push_back( os.str() + ": destination shape " + describeVec( dst ) + " expected " +
                             describeVec( want ) );

        std::set<std::vector<size_t>> images;
        for( size_t i = 0; i < src.at( 0 ); i++ )
        for( size_t j = 0; j < src.at( 1 ); j++ )
        {
            const std::vector<size_t> got = transformSideIndex( t, { i, j }, src );
            images.insert( got );
            for( size_t d = 0; d < got.size(); d++ )
                if( got.at( d ) >= dst.at( d ) )
                    wrong.push_back( os.str() + " at " + describeVec( { i, j } ) + ": image " +
                                     describeVec( got ) + " outside destination shape " + describeVec( dst ) );
        }
        if( images.size() != src.at( 0 ) * src.at( 1 ) )
            wrong.push_back( os.str() + ": not a bijection onto the destination lattice" );
    }

    INFO( "non-square transform problems: " << joined( wrong ) );
    CHECK( wrong.empty() );
}

TEST_CASE( "the shared transform's index and point maps describe the same map" )
{
    // Correction 4.  Deriving both from one record makes disagreement less
    // likely, not impossible - the two appliers can still read
    // source_axis_for_destination or source_axis_reversed differently - so this
    // stays PERMANENT after the refactor.  Non-square extents included, since
    // that is where a misread of which extent to use would show.
    const std::vector<size_t> src{ 4, 5 };
    const auto centreOf = []( const size_t i, const size_t n ) {
        return ( static_cast<double>( i ) + 0.5 ) / static_cast<double>( n );
    };
    std::vector<std::string> wrong;

    for( size_t sa = 0; sa < 6; sa++ )
    for( size_t sb = 0; sb < 6; sb++ )
    for( const TPPermutation p : { TPPermutation::ZeroToZero, TPPermutation::ZeroToOne,
                                   TPPermutation::ZeroToTwo, TPPermutation::ZeroToThree } )
    {
        const SideCoordinateTransform t = sideCoordinateTransform( 3, sa, sb, p );
        const std::vector<size_t> dst = transformedExtents( t, src );
        std::ostringstream os;
        os << "sides " << sa << " <-> " << sb << " perm " << static_cast<int>( p );
        for( size_t i = 0; i < src.at( 0 ); i++ )
        for( size_t j = 0; j < src.at( 1 ); j++ )
        {
            const std::vector<size_t> idx = transformSideIndex( t, { i, j }, src );
            const std::vector<double> pt =
                transformSidePoint( t, { centreOf( i, src.at( 0 ) ), centreOf( j, src.at( 1 ) ) } );
            for( size_t d = 0; d < idx.size(); d++ )
                if( std::abs( pt.at( d ) - centreOf( idx.at( d ), dst.at( d ) ) ) > 1e-13 )
                    wrong.push_back( os.str() + " at " + describeVec( { i, j } ) + " axis " +
                                     std::to_string( d ) + ": point and index disagree" );
        }
    }

    for( size_t sa = 0; sa < 4; sa++ )
    for( size_t sb = 0; sb < 4; sb++ )
    {
        const SideCoordinateTransform t = sideCoordinateTransform( 2, sa, sb, TPPermutation::Flip1d );
        for( size_t i = 0; i < n_elem; i++ )
        {
            const std::vector<size_t> idx = transformSideIndex( t, { i }, { n_elem } );
            const std::vector<double> pt = transformSidePoint( t, { centreOf( i, n_elem ) } );
            if( std::abs( pt.at( 0 ) - centreOf( idx.at( 0 ), n_elem ) ) > 1e-13 )
                wrong.push_back( "2d sides " + std::to_string( sa ) + " <-> " + std::to_string( sb ) +
                                 ": point and index disagree" );
        }
    }

    INFO( "index and point maps disagree on " << joined( wrong ) );
    CHECK( wrong.empty() );
}

TEST_CASE( "the shared transform validates its inputs" )
{
    CHECK_THROWS_AS( sideCoordinateTransform( 1, 0, 0, TPPermutation::Flip1d ), std::invalid_argument );
    CHECK_THROWS_AS( sideCoordinateTransform( 4, 0, 0, TPPermutation::ZeroToZero ), std::invalid_argument );
    CHECK_THROWS_AS( sideCoordinateTransform( 2, 4, 0, TPPermutation::Flip1d ), std::invalid_argument );
    CHECK_THROWS_AS( sideCoordinateTransform( 2, 0, 4, TPPermutation::Flip1d ), std::invalid_argument );
    CHECK_THROWS_AS( sideCoordinateTransform( 3, 6, 0, TPPermutation::ZeroToZero ), std::invalid_argument );
    CHECK_THROWS_AS( sideCoordinateTransform( 3, 0, 6, TPPermutation::ZeroToZero ), std::invalid_argument );
    // a permutation that cannot describe an interface of that dimension
    CHECK_THROWS_AS( sideCoordinateTransform( 2, 0, 1, TPPermutation::ZeroToZero ), std::invalid_argument );
    CHECK_THROWS_AS( sideCoordinateTransform( 3, 0, 1, TPPermutation::Flip1d ), std::invalid_argument );

    const SideCoordinateTransform t = sideCoordinateTransform( 3, 0, 1, TPPermutation::ZeroToZero );
    CHECK_THROWS_AS( transformedExtents( t, { 3 } ), std::invalid_argument );              // rank
    CHECK_THROWS_AS( transformedExtents( t, { 3, 0 } ), std::invalid_argument );           // positive extents
    CHECK_THROWS_AS( transformSideIndex( t, { 0 }, { 3 } ), std::invalid_argument );       // rank
    CHECK_THROWS_AS( transformSideIndex( t, { 3, 0 }, { 3, 3 } ), std::invalid_argument ); // index in range
    CHECK_THROWS_AS( transformSidePoint( t, { 0.5 } ), std::invalid_argument );            // rank

    // a hand-built record that is not a permutation of the tangential axes
    SideCoordinateTransform bad;
    bad.source_axis_for_destination.push_back( 0 );
    bad.source_axis_for_destination.push_back( 0 );
    bad.source_axis_reversed.push_back( false );
    bad.source_axis_reversed.push_back( false );
    CHECK_THROWS_AS( transformedExtents( bad, { 3, 3 } ), std::invalid_argument );
}

// ============ requirements on the trace and validator consumers ================
// These carried [!shouldfail] while the consumers were side-blind.  Stage 2
// removed the tag, not the cases: they guard the transform itself, below the
// spline fixtures.

TEST_CASE( "trace side index transform agrees with phi in 2d" )
{
    const auto patch = squarePatch();
    std::vector<std::string> wrong;
    for( size_t sa = 0; sa < 4; sa++ )
    {
        for( size_t sb = 0; sb < 4; sb++ )
        {
            const MultiPatchSplineSpace h1 =
                buildH1MultiPatchSplineSpace( { patch, patch }, conn2d( sa, sb, TPPermutation::Flip1d ) );
            const auto& mp = h1.basisComplex().parametricAtlas().cmap();
            bool consistent = false;
            const std::map<size_t, size_t> from_phi = phiCorr2d( mp, 0, sa, sb, consistent );
            const std::map<size_t, size_t> from_trace = traceCorr2d( h1, mp, sa, sb );
            std::ostringstream os;
            os << "sides " << sa << " <-> " << sb;
            if( not consistent ) wrong.push_back( os.str() + ": phi pairing inconsistent across darts" );
            else if( from_trace != from_phi )
                wrong.push_back( os.str() + ": phi" + describe( from_phi ) + "  trace" + describe( from_trace ) );
        }
    }
    INFO( "trace disagrees with phi on " << joined( wrong ) );
    CHECK( wrong.empty() );
}

TEST_CASE( "trace side index transform agrees with phi in 3d" )
{
    // ONE aggregate assertion.  Two independent constructions feed the same
    // list, so a partial fix cannot hide behind another still-failing row:
    //
    //  (1) EXHAUSTIVE, pure topology - all 144 (source side, destination side,
    //      permutation) triples, comparing permuteTraceSideIndex against phi
    //      directly.  The permutation is SPECIFIED here.
    //  (2) DERIVED, through the production dart-pair constructor and a real
    //      spline space, so the permutation comes from
    //      initializeInterMapConnections rather than from the fixture and the
    //      comparison runs through patchTraceMeshInterfaces, the real consumer.
    //
    // The per-triple classification counts appear in the failure message but are
    // deliberately NOT asserted: they describe the defect, so pinning them would
    // outlive the fix.
    std::vector<std::string> wrong;

    // ---------------- (1) exhaustive, pure ----------------
    const auto line = std::make_shared<const CombinatorialMap1d>( n_elem );
    const auto sq = std::make_shared<const TPCombinatorialMap>( line, line );
    const auto cube = std::make_shared<const TPCombinatorialMap>( sq, line );
    const std::vector<size_t> lengths{ n_elem, n_elem };

    size_t n_triples = 0, n_full_agree = 0, n_partial = 0, n_full_disagree = 0;
    for( size_t sa = 0; sa < 6; sa++ )
    for( size_t sb = 0; sb < 6; sb++ )
    for( const TPPermutation p : { TPPermutation::ZeroToZero, TPPermutation::ZeroToOne,
                                   TPPermutation::ZeroToTwo, TPPermutation::ZeroToThree } )
    {
        n_triples++;
        const ConstituentSide a{ 0, sa };
        const ConstituentSide b{ 1, sb };
        const MultiPatchCombinatorialMap mp( { cube, cube },
                                             InternalConnectionsMap{ { a, { p, b } }, { b, { p, a } } } );

        std::map<std::vector<size_t>, std::vector<size_t>> from_phi;
        iterateDartsWhile( *cube, [&]( const Dart& d ) {
            const auto n = topology::phi( mp, 3, mp.toGlobalDart( 0, d ) );
            if( not n.has_value() ) return true;
            const auto [q, nd] = mp.toLocalDart( *n );
            if( q != 1 ) return true;
            from_phi[tang3d( unflattenFull( *cube, d ).unflat_darts, sa )] =
                tang3d( unflattenFull( *cube, nd ).unflat_darts, sb );
            return true;
        } );

        std::ostringstream os;
        os << "sides " << sa << " <-> " << sb << " perm " << static_cast<int>( p );
        if( from_phi.size() != n_elem * n_elem )
        {
            wrong.push_back( os.str() + ": phi paired " + std::to_string( from_phi.size() ) +
                             " interface elements" );
            continue;
        }
        size_t agree = 0, dis = 0;
        for( const auto& [ta, tb] : from_phi )
            ( permuteTraceSideIndex( sideCoordinateTransform( 3, sa, sb, p ), ta, lengths ) == tb ) ? agree++ : dis++;

        if( dis == 0 ) n_full_agree++;
        else
        {
            ( agree == 0 ) ? n_full_disagree++ : n_partial++;
            wrong.push_back( os.str() + ": trace disagrees with phi on " + std::to_string( dis ) +
                             " of " + std::to_string( from_phi.size() ) + " interface elements" );
        }
    }
    if( n_triples != 144 ) wrong.push_back( "enumerated " + std::to_string( n_triples ) + " triples, expected 144" );

    // ---------------- (2) derived permutation, through the real consumer ----------------
    const KnotVector kv = uniformKV();
    const auto patch = makePatch( { kv, kv, kv }, { 2, 2, 2 } );
    const param::TPParametricAtlas& tp_atlas = patch->basisComplex().parametricAtlas();
    const auto tp_cmap = tp_atlas.cmapPtr();

    std::map<size_t, Dart> side_dart;
    iterateDartsWhile( *tp_cmap, [&]( const Dart& d ) {
        const auto s = faceSideId( tp_atlas, d );
        if( s.has_value() and not side_dart.count( *s ) ) side_dart.emplace( *s, d );
        return true;
    } );

    for( const auto [sa, sb] : { std::pair<size_t, size_t>{ 0, 1 }, std::pair<size_t, size_t>{ 5, 3 } } )
    {
        std::ostringstream os;
        os << "derived: sides " << sa << " <-> " << sb;
        if( not side_dart.count( sa ) or not side_dart.count( sb ) )
        {
            wrong.push_back( os.str() + ": no representative dart found" );
            continue;
        }

        const MultiPatchCombinatorialMap probe(
            { tp_cmap, tp_cmap }, { { { 0, side_dart.at( sa ) }, { 1, side_dart.at( sb ) } } } );
        const auto& conns = probe.connections();
        const auto it = conns.find( ConstituentSide{ 0, sa } );
        if( it == conns.end() or it->second.second.side_id != sb )
        {
            wrong.push_back( os.str() + ": constructor derived a different side pairing" );
            continue;
        }
        const TPPermutation perm = it->second.first;
        os << " (derived permutation " << static_cast<int>( perm ) << ")";

        const MultiPatchSplineSpace h1 = buildH1MultiPatchSplineSpace( { patch, patch }, conns );
        const auto& mp = h1.basisComplex().parametricAtlas().cmap();

        std::map<std::vector<size_t>, std::vector<size_t>> from_phi, from_trace;
        iterateDartsWhile( *tp_cmap, [&]( const Dart& d ) {
            if( faceSideId( tp_atlas, d ) != std::optional<size_t>( sa ) ) return true;
            const auto n = topology::phi( mp, 3, mp.toGlobalDart( 0, d ) );
            if( not n.has_value() ) return true;
            const auto [op, od] = mp.toLocalDart( *n );
            if( op != 1 ) return true;
            from_phi[tang3d( unflattenFull( *tp_cmap, d ).unflat_darts, sa )] =
                tang3d( unflattenFull( *tp_cmap, od ).unflat_darts, sb );
            return true;
        } );
        for( const TraceMeshInterface& iface : patchTraceMeshInterfaces( h1 ) )
        {
            if( not iface.second.has_value() ) continue;
            const auto [p1, d1] = mp.toLocalDart( iface.first.element.dart() );
            const auto [p2, d2] = mp.toLocalDart( iface.second->element.dart() );
            if( p1 != 0 or p2 != 1 ) continue;
            from_trace[tang3d( unflattenFull( *tp_cmap, d1 ).unflat_darts, sa )] =
                tang3d( unflattenFull( *tp_cmap, d2 ).unflat_darts, sb );
        }

        if( from_phi.empty() or from_trace.empty() )
        {
            wrong.push_back( os.str() + ": no cross-patch elements observed" );
            continue;
        }
        for( const auto& [key, traced] : from_trace )
        {
            const auto f = from_phi.find( key );
            if( f == from_phi.end() ) continue;
            const std::vector<size_t> spec = expected3d( key, sa, sb, perm, n_elem );
            if( traced != f->second or f->second != spec )
                wrong.push_back( os.str() + ": at " + describeVec( key ) +
                                 " phi " + describeVec( f->second ) +
                                 " trace " + describeVec( traced ) +
                                 " spec " + describeVec( spec ) );
        }
    }

    std::ostringstream summary;
    summary << n_triples << " exhaustive triples: " << n_full_agree << " fully agreeing, " << n_partial
            << " partially disagreeing, " << n_full_disagree << " fully disagreeing";
    INFO( "classification (reported, not asserted) - " << summary.str() );
    INFO( "trace disagrees with phi on " << joined( wrong ) );
    CHECK( wrong.empty() );
}

TEST_CASE( "strong-interface knot validation is side-aware" )
{
    // Sides 0 <-> 1 have an odd flip count, so the correct correspondence is
    // IDENTITY.  A patch glued to itself across an identical non-palindromic
    // interface therefore matches span for span and is COMPATIBLE, while the
    // mirrored fixture couples span 1 to span 2 and must be REJECTED.  The
    // former side-blind validator reached both opposite conclusions.
    std::vector<std::string> wrong;
    try
    {
        buildMirroredPair( false );
    }
    catch( const std::exception& e )
    {
        wrong.push_back( std::string( "B3 identical interface knots REJECTED: " ) + e.what() );
    }
    try
    {
        buildMirroredPair( true );
        wrong.push_back( "B4 mirrored interface knots ACCEPTED, but span 1 cannot couple to span 2" );
    }
    catch( const std::exception& ) {}
    INFO( "side-aware validation verdicts, " << joined( wrong ) );
    CHECK( wrong.empty() );
}
