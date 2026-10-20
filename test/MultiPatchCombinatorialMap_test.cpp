#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <TPCombinatorialMap.hpp>
#include <MultiPatchCombinatorialMap.hpp>
#include <CombinatorialMapMethods.hpp>
#include <stdexcept>
#include <Logging.hpp>

using namespace topology;
TEST_CASE( "Simple 2d multipatch cmap" )
{
    const auto cmap_1d_1 = std::make_shared<const CombinatorialMap1d>( 2 );
    const auto cmap_1d_2 = std::make_shared<const CombinatorialMap1d>( 1 );
    const auto cmap_tp_1 = std::make_shared<const TPCombinatorialMap>( cmap_1d_2, cmap_1d_1 );
    const auto cmap_tp_2 = std::make_shared<const TPCombinatorialMap>( cmap_1d_1, cmap_1d_2 );

    {
        const MultiPatchCombinatorialMap cmap( { cmap_tp_1, cmap_tp_2 }, { { { 0, Dart( 1 ) }, { 1, Dart( 4 ) } } } );

        CHECK( cellCount( cmap, 2 ) == 4 );
        CHECK( cellCount( cmap, 1 ) == 12 );
        CHECK( cellCount( cmap, 0 ) == 9 );

        const auto check_phi2 = [&]( const Dart& d, const Dart& expected ) {
            const auto phi2 = phi( cmap, 2, d );
            CHECK( phi2.has_value() );
            if( phi2.has_value() )
            {
                CHECK( phi2.value() == expected );
            }
        };
        check_phi2( 1, 12 );
        check_phi2( 5, 8 );
        check_phi2( 8, 5 );
        check_phi2( 12, 1 );

        CHECK( not phi( cmap, 2, Dart( 0 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 3 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 6 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 7 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 10 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 11 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 13 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 14 ) ).has_value() );
    }

    {
        const MultiPatchCombinatorialMap cmap( { cmap_tp_1, cmap_tp_2 }, { { { 0, Dart( 1 ) }, { 1, Dart( 2 ) } } } );

        CHECK( cellCount( cmap, 2 ) == 4 );
        CHECK( cellCount( cmap, 1 ) == 12 );
        CHECK( cellCount( cmap, 0 ) == 9 );

        const auto check_phi2 = [&]( const Dart& d, const Dart& expected ) {
            const auto phi2 = phi( cmap, 2, d );
            CHECK( phi2.has_value() );
            if( phi2.has_value() )
            {
                CHECK( phi2.value() == expected );
            }
        };
        check_phi2( 1, 10 );
        check_phi2( 5, 14 );
        check_phi2( 14, 5 );
        check_phi2( 10, 1 );

        CHECK( not phi( cmap, 2, Dart( 0 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 3 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 6 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 7 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 8 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 11 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 12 ) ).has_value() );
        CHECK( not phi( cmap, 2, Dart( 13 ) ).has_value() );
    }
}

TEST_CASE( "Simple 3d multipatch cmap" )
{
    const auto cmap_1d_1 = std::make_shared<const CombinatorialMap1d>( 1 );
    const auto cmap_1d_2 = std::make_shared<const CombinatorialMap1d>( 2 );
    const auto cmap_2d_1 = std::make_shared<const TPCombinatorialMap>( cmap_1d_2, cmap_1d_1 );
    const auto cmap_2d_2 = std::make_shared<const TPCombinatorialMap>( cmap_1d_1, cmap_1d_2 );
    const auto cmap_tp_1 = std::make_shared<const TPCombinatorialMap>( cmap_2d_1, cmap_1d_1 );
    const auto cmap_tp_2 = std::make_shared<const TPCombinatorialMap>( cmap_2d_2, cmap_1d_1 );

    const auto check_phi3 = [&]( const CombinatorialMap& cmap, const Dart& d, const Dart& expected ) {
        const auto phi3 = phi( cmap, 3, d );
        CHECK( phi3.has_value() );
        if( phi3.has_value() )
        {
            CHECK( phi3.value() == expected );
        }
    };

    {
        const MultiPatchCombinatorialMap cmap( { cmap_tp_1, cmap_tp_2 }, { { { 0, Dart( 1 ) }, { 1, Dart( 19 ) } } } );

        CHECK( cellCount( cmap, 3 ) == 4 );
        CHECK( cellCount( cmap, 2 ) == 20 );
        CHECK( cellCount( cmap, 1 ) == 33 );
        CHECK( cellCount( cmap, 0 ) == 18 );

        check_phi3( cmap, 1, 67 );
        check_phi3( cmap, 2, 70 );
        check_phi3( cmap, 3, 69 );
        check_phi3( cmap, 4, 68 );
        check_phi3( cmap, 25, 91 );
        check_phi3( cmap, 26, 94 );
        check_phi3( cmap, 27, 93 );
        check_phi3( cmap, 28, 92 );

        CHECK( not phi( cmap, 3, Dart( 0 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 20 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 13 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 37 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 23 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 47 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 31 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 24 ) ).has_value() );

        CHECK( not phi( cmap, 3, Dart( 48 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 55 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 51 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 71 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 88 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 79 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 72 ) ).has_value() );
        CHECK( not phi( cmap, 3, Dart( 89 ) ).has_value() );
    }

    {
        const MultiPatchCombinatorialMap cmap( { cmap_tp_1, cmap_tp_2 }, { { { 0, Dart( 1 ) }, { 1, Dart( 9 ) } } } );

        CHECK( cellCount( cmap, 3 ) == 4 );
        CHECK( cellCount( cmap, 2 ) == 20 );
        CHECK( cellCount( cmap, 1 ) == 33 );
        CHECK( cellCount( cmap, 0 ) == 18 );

        check_phi3( cmap, 1, 57 );
        check_phi3( cmap, 2, 56 );
        check_phi3( cmap, 3, 55 );
        check_phi3( cmap, 4, 58 );
        size_t n_bdry_darts = 0;
        iterateDartsWhile( cmap, [&]( const Dart& d ) {
            if( onBoundary( cmap, d ) ) n_bdry_darts++;
            return true;
        } );
        CHECK( n_bdry_darts == 64 );
    }

    {
        const MultiPatchCombinatorialMap cmap( { cmap_tp_1, cmap_tp_1 }, { { { 0, Dart( 1 ) }, { 1, Dart( 25 ) } } } );

        CHECK( cellCount( cmap, 3 ) == 4 );
        CHECK( cellCount( cmap, 2 ) == 20 );
        CHECK( cellCount( cmap, 1 ) == 33 );
        CHECK( cellCount( cmap, 0 ) == 18 );

        check_phi3( cmap, 1, 73 );
        check_phi3( cmap, 2, 76 );
        check_phi3( cmap, 3, 75 );
        check_phi3( cmap, 4, 74 );

        check_phi3( cmap, 25, 49 );
        check_phi3( cmap, 26, 52 );
        check_phi3( cmap, 27, 51 );
        check_phi3( cmap, 28, 50 );

        size_t n_bdry_darts = 0;
        iterateDartsWhile( cmap, [&]( const Dart& d ) {
            if( onBoundary( cmap, d ) ) n_bdry_darts++;
            return true;
        } );
        CHECK( n_bdry_darts == 64 );
    }
}
TEST_CASE( "A patch side has at most one whole-side partner" )
{
    const auto line = std::make_shared<const CombinatorialMap1d>( 1 );
    const auto quad = std::make_shared<const TPCombinatorialMap>( line, line );
    const auto hex = std::make_shared<const TPCombinatorialMap>( quad, line );

    // Side 0 of a hex carries local darts 7, 8, 9 and 10, so these two
    // connections assign patch 0 side 0 two different partners.  Rejected at
    // construction rather than silently dropped.
    const std::map<std::pair<size_t, Dart>, std::pair<size_t, Dart>> conflicting{
        { { 0, Dart( 7 ) }, { 1, Dart( 19 ) } }, { { 0, Dart( 8 ) }, { 2, Dart( 19 ) } } };

    bool threw_for_second_partner = false;
    try
    {
        MultiPatchCombinatorialMap( { hex, hex, hex }, conflicting );
    }
    catch( const std::exception& e )
    {
        threw_for_second_partner =
            std::string( e.what() ).find( "at most one whole-side partner" ) != std::string::npos;
    }
    CHECK( threw_for_second_partner );

    // Stating one gluing from both ends is the same gluing, not a second
    // partner: the permutation is symmetric in the two sides, so this stays
    // accepted and yields the two stored directions.
    const std::map<std::pair<size_t, Dart>, std::pair<size_t, Dart>> restated{
        { { 0, Dart( 7 ) }, { 1, Dart( 19 ) } }, { { 1, Dart( 19 ) }, { 0, Dart( 7 ) } } };

    const MultiPatchCombinatorialMap cmap( { hex, hex }, restated );
    CHECK( cmap.connections().size() == 2 );
}

TEST_CASE( "interfaceAtDart validates patch-boundary membership, not element-local direction" )
{
    const auto line1 = std::make_shared<const CombinatorialMap1d>( 1 );
    const auto line2 = std::make_shared<const CombinatorialMap1d>( 2 );
    const auto quad = std::make_shared<const TPCombinatorialMap>( line1, line1 );
    const auto hex = std::make_shared<const TPCombinatorialMap>( quad, line1 );
    const auto refined_quad = std::make_shared<const TPCombinatorialMap>( line2, line1 );
    const auto refined = std::make_shared<const TPCombinatorialMap>( refined_quad, line1 );

    // Discover one patch-boundary dart per side through the public API alone:
    // glue a candidate to a lone hex and read back which side was registered.
    std::map<size_t, Dart> boundary_dart_for_side;
    iterateDartsWhile( *refined, [&]( const Dart& d ) {
        if( phi( *refined, 3, d ).has_value() ) return true;
        try
        {
            const MultiPatchCombinatorialMap probe( { refined, hex }, { { { 0, d }, { 1, Dart( 19 ) } } } );
            for( const auto& [side, conn] : probe.connections() )
                if( side.constituent_id == 0 ) boundary_dart_for_side.emplace( side.side_id, d );
        }
        catch( const std::exception& )
        {
        }
        return true;
    } );
    REQUIRE( boundary_dart_for_side.count( 0 ) == 1 );
    REQUIRE( boundary_dart_for_side.count( 1 ) == 1 );

    // Patch 0 has two elements along its first axis, so the face BETWEEN them
    // is normal to that axis.  With BOTH sides normal to that axis glued, every
    // interior face of patch 0 shares its element-local direction with a glued
    // side, and a lookup keyed on direction alone would hand each of them an
    // exterior side's interface.
    const MultiPatchCombinatorialMap cmap( { refined, hex, hex },
                                           { { { 0, boundary_dart_for_side.at( 0 ) }, { 1, Dart( 19 ) } },
                                             { { 0, boundary_dart_for_side.at( 1 ) }, { 2, Dart( 19 ) } } } );

    size_t n_interior = 0, n_interior_refused = 0, n_glued_resolved = 0, n_unglued_refused = 0, n_other = 0;
    iterateDartsWhile( *refined, [&]( const Dart& d ) {
        const bool interior = phi( *refined, 3, d ).has_value();
        if( interior ) n_interior++;
        try
        {
            const PatchSideInterface iface = interfaceAtDart( cmap, cmap.toGlobalDart( 0, d ) );
            if( interior or iface.source.constituent_id != 0 ) n_other++;
            else n_glued_resolved++;
        }
        catch( const std::exception& e )
        {
            const std::string what( e.what() );
            if( interior and what.find( "interior face" ) != std::string::npos ) n_interior_refused++;
            else if( not interior and what.find( "unglued exterior side" ) != std::string::npos )
                n_unglued_refused++;
            else n_other++;
        }
        return true;
    } );

    CHECK( n_interior > 0 );              // the hazard is actually present
    CHECK( n_interior_refused == n_interior );
    CHECK( n_glued_resolved > 0 );        // glued boundary faces still resolve
    CHECK( n_unglued_refused > 0 );       // the four unglued sides are named as such
    CHECK( n_other == 0 );

    // An interface must be the same interface read from either end.
    const Dart interface_dart = cmap.toGlobalDart( 0, boundary_dart_for_side.at( 0 ) );
    const PatchSideInterface fwd = interfaceAtDart( cmap, interface_dart );
    CHECK( fwd.source.constituent_id == 0 );
    CHECK( fwd.destination.constituent_id == 1 );
    const PatchSideInterface rev = interfaceAtDart( cmap, phi( cmap, 3, interface_dart ).value() );
    CHECK( rev.source.constituent_id == fwd.destination.constituent_id );
    CHECK( rev.source.side_id == fwd.destination.side_id );
    CHECK( rev.destination.side_id == fwd.source.side_id );
    CHECK( rev.permutation == fwd.permutation );
}

TEST_CASE( "The direct connection-map constructor refuses malformed connectivity" )
{
    using Catch::Matchers::ContainsSubstring;
    using Catch::Matchers::MessageMatches;

    const auto line = std::make_shared<const CombinatorialMap1d>( 1 );
    const auto quad = std::make_shared<const TPCombinatorialMap>( line, line );
    const auto hex = std::make_shared<const TPCombinatorialMap>( quad, line );
    const std::vector<std::shared_ptr<const TPCombinatorialMap>> three_hexes{ hex, hex, hex };

    using ConstituentSide = MultiPatchCombinatorialMap::ConstituentSide;
    using TPPermutation = MultiPatchCombinatorialMap::TPPermutation;
    using Connections = MultiPatchCombinatorialMap::InternalConnectionsMap;
    const auto perm = TPPermutation::ZeroToZero;
    const auto other_perm = TPPermutation::ZeroToTwo;

    const ConstituentSide a{ 0, 0 };
    const ConstituentSide b{ 1, 1 };
    const ConstituentSide c{ 2, 1 };

    // The derived constructor writes both directions with one permutation; the
    // direct one has to supply exactly that, so the baseline must be accepted.
    REQUIRE_NOTHROW( MultiPatchCombinatorialMap( three_hexes, Connections{ { a, { perm, b } }, { b, { perm, a } } } ) );

    // Each refusal must name the offending connection, source and destination,
    // and the reason.  The expected text is written out by hand so a change to
    // the rendering cannot also change what the test expects.
    const auto refuses = []( const std::vector<std::shared_ptr<const TPCombinatorialMap>>& patches,
                             const Connections& connections,
                             const std::string& offender,
                             const std::string& reason ) {
        CHECK_THROWS_MATCHES( MultiPatchCombinatorialMap( patches, connections ),
                              std::runtime_error,
                              MessageMatches( ContainsSubstring( "Malformed patch connection " + offender ) and
                                              ContainsSubstring( reason ) ) );
    };

    // A missing reverse entry.
    refuses( three_hexes, { { a, { perm, b } } }, "{0, 0} -> {1, 1}", "registers no connection" );

    // A -> B alongside C -> B: B names only A back, so C's entry is the one
    // refused.  A map keyed by source cannot see this.
    refuses( three_hexes,
             { { a, { perm, b } }, { c, { perm, b } }, { b, { perm, a } } },
             "{2, 1} -> {1, 1}",
             "different side" );

    // A reverse entry naming a different permutation.
    refuses( three_hexes,
             { { a, { perm, b } }, { b, { other_perm, a } } },
             "{0, 0} -> {1, 1}",
             "different permutation" );

    // A reverse entry naming a different side in return.
    refuses( three_hexes,
             { { a, { perm, b } }, { b, { perm, c } }, { c, { perm, b } } },
             "{0, 0} -> {1, 1}",
             "different side" );

    // A patch id outside the constituent list.
    refuses( { hex, hex }, { { a, { perm, c } }, { c, { perm, a } } }, "{0, 0} -> {2, 1}", "patch id is outside" );

    // A side id outside the patch dimension, refused by sideCoordinateTransform.
    const ConstituentSide off_patch{ 1, 6 };
    refuses( three_hexes,
             { { a, { perm, off_patch } }, { off_patch, { perm, a } } },
             "{0, 0} -> {1, 6}",
             "outside the patch dimension" );

    // The two-dimensional permutation on a three-dimensional interface.
    refuses( three_hexes,
             { { a, { TPPermutation::Flip1d, b } }, { b, { TPPermutation::Flip1d, a } } },
             "{0, 0} -> {1, 1}",
             "requires a face permutation" );

    // A patch side glued to itself.
    refuses( three_hexes, { { a, { perm, a } } }, "{0, 0} -> {0, 0}", "cannot be glued to itself" );
}

TEST_CASE( "Both multipatch constructors refuse the same malformed constituent lists" )
{
    using Catch::Matchers::ContainsSubstring;
    using Catch::Matchers::MessageMatches;

    const auto line = std::make_shared<const CombinatorialMap1d>( 1 );
    const auto quad = std::make_shared<const TPCombinatorialMap>( line, line );
    const auto hex = std::make_shared<const TPCombinatorialMap>( quad, line );

    using Patches = std::vector<std::shared_ptr<const TPCombinatorialMap>>;
    using Connections = MultiPatchCombinatorialMap::InternalConnectionsMap;
    using DartConnections = std::map<std::pair<size_t, Dart>, std::pair<size_t, Dart>>;

    // Every case is checked through both public paths with NO connections, so
    // nothing per-connection can be what refuses it.
    const auto both_refuse = []( const Patches& patches, const std::string& reason ) {
        CHECK_THROWS_MATCHES( MultiPatchCombinatorialMap( patches, Connections{} ),
                              std::runtime_error,
                              MessageMatches( ContainsSubstring( reason ) ) );
        CHECK_THROWS_MATCHES( MultiPatchCombinatorialMap( patches, DartConnections{} ),
                              std::runtime_error,
                              MessageMatches( ContainsSubstring( reason ) ) );
    };

    both_refuse( {}, "Cannot create multipatch of zero patches" );

    // A null patch would otherwise be dereferenced while building dart ranges.
    both_refuse( { hex, nullptr }, "Multipatch constituent 1 is a null patch" );
    both_refuse( { nullptr, hex }, "Multipatch constituent 0 is a null patch" );

    // A disconnected mixed-dimension collection: no connection crosses the two
    // dimensions, and dim() would report only the first patch's.
    both_refuse( { hex, quad }, "Multipatch constituent 1 has dimension 2, but constituent 0 has dimension 3" );
    both_refuse( { quad, hex, quad }, "Multipatch constituent 1 has dimension 3, but constituent 0 has dimension 2" );
}
