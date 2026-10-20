#include <catch2/catch_test_macros.hpp>
#include <HierarchicalTPSplineSpace.hpp>
#include <CombinatorialMapMethods.hpp>
#include <Logging.hpp>
#include <CommonUtils.hpp>
#include <VTKOutput.hpp>
#include <SplineSpaceEvaluator.hpp>

using namespace topology;
using namespace param;
using namespace basis;

constexpr bool VTK_OUTPUT = false;

TEST_CASE( "Simple hierarchical atlas 1" )
{
    const KnotVector kv1( {0, 0, 0, 1, 1, 1}, 1e-10 );
    const KnotVector kv2( {0, 0, 0, 0.5, 1, 1, 1}, 1e-10 );
    const KnotVector kv4( {0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1}, 1e-10 );

    const size_t degree = 2;
    const auto tp_ss_1 = std::make_shared<const TPSplineSpace>( buildBSpline( { kv1, kv2 }, { degree, degree } ) );
    const auto tp_ss_2 = std::make_shared<const TPSplineSpace>( buildBSpline( { kv2, kv4 }, { degree, degree } ) );

    const HierarchicalTPSplineSpace ss = buildHierarchicalSplineSpace( { tp_ss_1, tp_ss_2 }, {
        { Face( Dart( 4 ) ) },
        { Face( Dart( 0 ) ), Face( Dart( 4 ) ), Face( Dart( 8 ) ), Face( Dart( 12 ) ) }
    } );

    const Eigen::MatrixX3d geom = ( Eigen::MatrixX3d( 17, 3 ) <<
        0, 0.5, 0,
        0.5, 0.5, 0,
        1, 0.5, 0,
        0, 1.5, 0,
        0.5, 1.5, 0,
        1, 1.5, 0,
        0, 2, 0,
        0.5, 2, 0,
        1, 2, 0,
        0, 0, 0,
        0.25, 0, 0,
        0.75, 0, 0,
        1, 0, 0,
        0, 0.25, 0,
        0.25, 0.25, 0,
        0.75, 0.25, 0,
        1, 0.25, 0 ).finished();

    if( VTK_OUTPUT ) io::outputBezierMeshToVTK( ss, geom, "hier_bez_test.vtu" );

    eval::SplineSpaceEvaluator evaler( ss, 0 );

    const auto& atlas = ss.basisComplex().parametricAtlas();
    const auto test_center_pt = [&]( const Face& f, const Eigen::Vector3d& expected ) {
        const ParentPoint ppt = pointOnBoundary( atlas.parentDomain( f ), parentDomainBoundary( atlas, f ) );
        evaler.localizeElement( f );
        evaler.localizeParentPoint( ppt );

        const Eigen::Vector3d pt = evaler.evaluateManifold( geom.transpose() );

        CHECK( util::equals( pt, expected, 1e-9 ) );
    };

    test_center_pt( Face( 5 ), Eigen::Vector3d( 0.5, 1.5, 0.0 ) );
    test_center_pt( Face( 8 ), Eigen::Vector3d( 0.25, 0.25, 0.0 ) );
    test_center_pt( Face( 12 ), Eigen::Vector3d( 0.75, 0.25, 0.0 ) );
}

TEST_CASE( "Simple hierarchical atlas 2" )
{
    const KnotVector kv1( {0, 0, 1, 1}, 1e-10 );
    const KnotVector kv2( {0, 0, 0.5, 1, 1}, 1e-10 );
    const KnotVector kv4( {0, 0, 0.25, 0.5, 0.75, 1, 1}, 1e-10 );
    const KnotVector kv8( { 0, 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1 }, 1e-10 );

    const size_t degree = 1;
    const auto tp_ss_1 = std::make_shared<const TPSplineSpace>( buildBSpline( { kv1, kv2 }, { degree, degree } ) );
    const auto tp_ss_2 = std::make_shared<const TPSplineSpace>( buildBSpline( { kv2, kv4 }, { degree, degree } ) );
    const auto tp_ss_3 = std::make_shared<const TPSplineSpace>( buildBSpline( { kv4, kv8 }, { degree, degree } ) );

    const HierarchicalTPSplineSpace ss = buildHierarchicalSplineSpace( { tp_ss_1, tp_ss_2, tp_ss_3 }, {
        { Face( Dart( 4 ) ) },
        { Face( Dart( 0 ) ), Face( Dart( 4 ) ), Face( Dart( 8 ) ) },
        { Face( Dart( 40 ) ), Face( Dart( 44 ) ), Face( Dart( 56 ) ), Face( Dart( 60 ) ) }
    } );

    const Eigen::MatrixX3d geom = ( Eigen::MatrixX3d( 12, 3 ) <<
        0, 1, 0,
        1, 1, 0,
        0, 2, 0,
        1, 2, 0,
        0, 0, 0,
        0.5, 0, 0,
        1, 0, 0,
        0, 0.5, 0,
        0.5, 0.5, 0,
        1, 0.5, 0,
        0.75, 0.75, 0,
        1.0, 0.75, 0 ).finished();

    if( VTK_OUTPUT ) io::outputBezierMeshToVTK( ss, geom, "hier_bez_test.vtu" );

    eval::SplineSpaceEvaluator evaler( ss, 0 );

    const auto& atlas = ss.basisComplex().parametricAtlas();
    const auto test_center_pt = [&]( const Face& f, const Eigen::Vector3d& expected ) {
        const ParentPoint ppt = pointOnBoundary( atlas.parentDomain( f ), parentDomainBoundary( atlas, f ) );
        evaler.localizeElement( f );
        evaler.localizeParentPoint( ppt );

        const Eigen::Vector3d pt = evaler.evaluateManifold( geom.transpose() );

        CHECK( util::equals( pt, expected, 1e-9 ) );
    };

    test_center_pt( Face( 5 ), Eigen::Vector3d( 0.5, 1.5, 0.0 ) );
    test_center_pt( Face( 8 ), Eigen::Vector3d( 0.25, 0.25, 0.0 ) );
    test_center_pt( Face( 12 ), Eigen::Vector3d( 0.75, 0.25, 0.0 ) );
    test_center_pt( Face( 80 ), Eigen::Vector3d( 0.625, 0.625, 0.0 ) );
}

TEST_CASE( "Simple hierarchical atlas 3" )
{
    const KnotVector kv1( { 0, 0, 1, 1}, 1e-10 );
    const KnotVector kv2( { 0, 0, 1, 2, 2}, 1e-10 );
    const KnotVector kv3 = nAdicRefine( kv1, 3 );
    const KnotVector kv6 = nAdicRefine( kv2, 3 );

    const size_t degree = 1;
    const auto tp_ss_1 = std::make_shared<const TPSplineSpace>( buildBSpline( { kv1, kv2 }, { degree, degree } ) );
    const auto tp_ss_2 = std::make_shared<const TPSplineSpace>( buildBSpline( { kv3, kv6 }, { degree, degree } ) );

    const HierarchicalTPSplineSpace ss = buildHierarchicalSplineSpace( { tp_ss_1, tp_ss_2 }, {
        { Face( Dart( 4 ) ) },
        { Face( Dart( 0 ) ), Face( Dart( 4 ) ), Face( Dart( 8 ) ), Face( Dart( 12 ) ), Face( 16 ), Face( 20 ), Face( 24 ), Face( 28 ), Face( 32 ) }
    } );

    const Eigen::MatrixX3d geom = ( Eigen::MatrixX3d( 16, 3 ) <<
        0, 1, 0,
        1, 1, 0,
        0, 2, 0,
        1, 2, 0,
        0, 0, 0,
        0.3333, 0, 0,
        0.6666, 0, 0,
        1, 0, 0,
        0, 0.3333, 0,
        0.3333, 0.3333, 0,
        0.6666, 0.3333, 0,
        1, 0.3333, 0,
        0, 0.6666, 0,
        0.3333, 0.6666, 0,
        0.6666, 0.6666, 0,
        1, 0.6666, 0 ).finished();

    if( VTK_OUTPUT ) io::outputBezierMeshToVTK( ss, geom, "hier_bez_test.vtu" );

    eval::SplineSpaceEvaluator evaler( ss, 0 );

    const auto& atlas = ss.basisComplex().parametricAtlas();
    const auto test_center_pt = [&]( const Face& f, const Eigen::Vector3d& expected ) {
        const ParentPoint ppt = pointOnBoundary( atlas.parentDomain( f ), parentDomainBoundary( atlas, f ) );
        evaler.localizeElement( f );
        evaler.localizeParentPoint( ppt );

        const Eigen::Vector3d pt = evaler.evaluateManifold( geom.transpose() );
        std::cout << pt.transpose() << std::endl;

        CHECK( util::equals( pt, expected, 1e-4 ) );
    };

    test_center_pt( Face( 5 ), Eigen::Vector3d( 0.5, 1.5, 0.0 ) );
    test_center_pt( Face( 8 ), Eigen::Vector3d( 1.0/6.0, 1.0/6.0, 0.0 ) );
    test_center_pt( Face( 12 ), Eigen::Vector3d( 0.5, 1.0/6.0, 0.0 ) );
    test_center_pt( Face( 32 ), Eigen::Vector3d( 1.0/6.0, 5.0/6.0, 0.0 ) );

    iterateCellsWhile( ss.basisComplex().parametricAtlas().cmap(), 2, [&]( const Face f ){
        std::cout << f << std::endl << ss.connectivity( f ) << std::endl << ss.extractionOperator( f ) << std::endl << std::endl;
        return true;
    } );
}

TEST_CASE( "Simple 3d hierarchical atlas 1" )
{
    const KnotVector kv1( {0, 0, 0, 1, 1, 1}, 1e-10 );
    const KnotVector kv2( {0, 0, 0, 0.5, 1, 1, 1}, 1e-10 );
    const KnotVector kv4( {0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1}, 1e-10 );

    const size_t degree = 2;
    const auto tp_ss_1 = std::make_shared<const TPSplineSpace>( buildBSpline( { kv1, kv1, kv2 }, { degree, degree, degree } ) );
    const auto tp_ss_2 = std::make_shared<const TPSplineSpace>( buildBSpline( { kv2, kv2, kv4 }, { degree, degree, degree } ) );

    const HierarchicalTPSplineSpace ss = buildHierarchicalSplineSpace( { tp_ss_1, tp_ss_2 }, {
        { Volume( 24 ) },
        { Volume( 0 ), Volume( 24 ), Volume( 48 ), Volume( 72 ), Volume( 96 ), Volume( 120 ), Volume( 144 ), Volume( 168 ) }
    } );

    Eigen::MatrixX3d geom( ss.numFunctions(), 3 );
    const std::vector<Eigen::MatrixXd> geoms = { util::tensorProduct( { grevillePoints( kv1, degree ), grevillePoints( kv1, degree ), grevillePoints( kv2, degree ) } ),
                                                 util::tensorProduct( { grevillePoints( kv2, degree ), grevillePoints( kv2, degree ), grevillePoints( kv4, degree ) } ) };
    size_t ii = 0;
    size_t j = 0;
    for( const auto& active_funcs : ss.activeFunctions() )
    {
        for( const auto& f : active_funcs )
        {
            geom.row( ii++ ) = geoms.at( j ).row( f );
        }
        j++;
    }

    if( VTK_OUTPUT ) io::outputBezierMeshToVTK( ss, geom, "hier_bez_test.vtu" );

    eval::SplineSpaceEvaluator evaler( ss, 0 );

    const auto& atlas = ss.basisComplex().parametricAtlas();
    const auto test_center_pt = [&]( const Volume& v, const Eigen::Vector3d& expected ) {
        const ParentPoint ppt = pointOnBoundary( atlas.parentDomain( v ), parentDomainBoundary( atlas, v ) );
        evaler.localizeElement( v );
        evaler.localizeParentPoint( ppt );

        const Eigen::Vector3d pt = evaler.evaluateManifold( geom.transpose() );

        CHECK( util::equals( pt, expected, 1e-9 ) );
    };

    test_center_pt( Volume( 26 ), Eigen::Vector3d( 0.5, 0.5, 0.75 ) );
    test_center_pt( Volume( 48 ), Eigen::Vector3d( 0.25, 0.25, 0.125 ) );
    test_center_pt( Volume( 72 ), Eigen::Vector3d( 0.75, 0.25, 0.125 ) );
    test_center_pt( Volume( 96 ), Eigen::Vector3d( 0.25, 0.75, 0.125 ) );
    test_center_pt( Volume( 120 ), Eigen::Vector3d( 0.75, 0.75, 0.125 ) );
    test_center_pt( Volume( 144 ), Eigen::Vector3d( 0.25, 0.25, 0.375 ) );
    test_center_pt( Volume( 168 ), Eigen::Vector3d( 0.75, 0.25, 0.375 ) );
    test_center_pt( Volume( 192 ), Eigen::Vector3d( 0.25, 0.75, 0.375 ) );
    test_center_pt( Volume( 216 ), Eigen::Vector3d( 0.75, 0.75, 0.375 ) );
}

TEST_CASE( "Simple 3d hierarchical spline space 2" )
{
    const KnotVector kv1( {0, 0, 0, 1, 1, 1}, 1e-10 );
    const KnotVector kv2( {0, 0, 0, 0.5, 1, 1, 1}, 1e-10 );
    const KnotVector kv4( {0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1}, 1e-10 );

    const size_t degree = 2;
    const auto tp_ss_1 = std::make_shared<const TPSplineSpace>( buildBSpline( { kv1, kv2, kv2 }, { degree, degree, degree } ) );
    const auto tp_ss_2 = std::make_shared<const TPSplineSpace>( buildBSpline( { kv2, kv4, kv4 }, { degree, degree, degree } ) );

    const HierarchicalTPSplineSpace ss = buildHierarchicalSplineSpace( { tp_ss_1, tp_ss_2 }, {
        { Volume( 24 ), Volume( 48 ), Volume( 72 ) },
        { Volume( 0 ), Volume( 24 ), Volume( 48 ), Volume( 72 ), Volume( 192 ), Volume( 216 ), Volume( 240 ), Volume( 264 ) }
    } );

    CHECK( cellCount( ss.basisComplex().parametricAtlas().cmap(), 3 ) == 11 );
    CHECK( cellCount( ss.basisComplex().parametricAtlas().cmap(), 2 ) == 50 );
    CHECK( cellCount( ss.basisComplex().parametricAtlas().cmap(), 1 ) == 75 );
    CHECK( cellCount( ss.basisComplex().parametricAtlas().cmap(), 0 ) == 37 );
    CHECK( ss.numFunctions() == 61 );

    iterateCellsWhile( ss.basisComplex().parametricAtlas().cmap(), 3, [&]( const Volume& v ) {
        const auto [level, dart] = ss.basisComplex().parametricAtlas().cmap().unrefinedAncestorDartOfCell( v );
        if( level == 0 ) CHECK( ss.connectivity( v ).size() == 27 );
        else CHECK( ss.connectivity( v ).size() >= 27 );
        CHECK( util::equals( ss.extractionOperator( v ).colwise().sum().transpose(), Eigen::VectorXd::Ones( 27 ), 1e-9 ) );
        return true;
    } );

    const Eigen::MatrixX3d geom = grevillePoints( ss );

    if( VTK_OUTPUT )
    {
        io::outputBezierMeshToVTK( ss, geom, "hier_3d_bez_test.vtu" );

        SimplicialComplex cpts;
        for( size_t i = 0; i < geom.rows(); i++ )
        {
            cpts.points.push_back( geom.row( i ) );
            cpts.simplices.push_back( { i } );
        }
        io::VTKOutputObject out( cpts );
        io::outputSimplicialFieldToVTK( out, "control_points.vtu" );
    }
}

// Item 1, step 5 (outstanding_issues.txt): a leaf-level jump of two inside one patch.
TEST_CASE( "Three levels, jump 2 inside a patch: THB active functions" )
{
    // s in [0,2] halved per level, t in [0,1] fixed, degree 2.  Leaves: L0 [0,1], L1 [1.5,2],
    // L2 [1,1.25] and [1.25,1.5]; so Omega^1 = [1,2] and Omega^2 = [1,1.5].
    const KnotVector kv_t( { 0, 0, 0, 1, 1, 1 }, 1e-10 );
    const std::vector<std::vector<size_t>> leaf_s = { { 0 }, { 3 }, { 4, 5 } };
    std::vector<std::shared_ptr<const TPSplineSpace>> levels;
    std::vector<std::vector<Cell>> leaves( leaf_s.size() );
    for( size_t level = 0; level < leaf_s.size(); level++ )
    {
        const size_t n = 2 << level;
        std::vector<double> knots = { 0, 0 };
        for( size_t i = 0; i <= n; i++ ) knots.push_back( 2.0 * i / n );
        knots.insert( knots.end(), { 2, 2 } );
        levels.push_back(
            std::make_shared<const TPSplineSpace>( buildBSpline( { KnotVector( knots, 1e-10 ), kv_t }, { 2, 2 } ) ) );
        const TPCombinatorialMap& level_cmap = levels.back()->basisComplex().parametricAtlas().cmap();
        for( const size_t s : leaf_s.at( level ) )
            leaves.at( level ).emplace_back(
                flattenFull( level_cmap, FullyUnflattenedDart( { Dart( s ), Dart( 0 ) } ) ), 2 );
    }
    const HierarchicalTPSplineSpace ss = buildHierarchicalSplineSpace( levels, leaves );

    // Level function id s + n_s * t.  Active iff supp in Omega^l and not in Omega^(l+1):
    // L0 s = 0, 1, 2 (s = 3 has supp [1,2]); L1 s = 4, 5 (s = 3 has supp [0.5,2]); none on L2.
    const std::vector<std::vector<size_t>> expected = { { 0, 1, 2, 4, 5, 6, 8, 9, 10 }, { 4, 5, 10, 11, 16, 17 }, {} };
    std::vector<std::vector<size_t>> active;
    for( const auto& level_funcs : ss.activeFunctions() ) active.emplace_back( level_funcs.begin(), level_funcs.end() );
    CHECK( active == expected );
}
