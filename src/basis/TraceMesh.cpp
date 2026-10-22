#include <TraceMesh.hpp>
#include <BasisComplex.hpp>
#include <CombinatorialMapMethods.hpp>
#include <HierarchicalMultiPatchCombinatorialMap.hpp>
#include <HierarchicalMultiPatchParametricAtlas.hpp>
#include <IndexOperations.hpp>
#include <MultiPatchParametricAtlas.hpp>
#include <ParametricAtlas.hpp>
#include <TPCombinatorialMap.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>

namespace basis
{
    namespace
    {
        using TPPermutation = topology::MultiPatchCombinatorialMap::TPPermutation;
        using SideIndex = std::vector<size_t>;

        struct LevelTraceMesh
        {
            size_t level = 0;
            const topology::MultiPatchCombinatorialMap* cmap = nullptr;
            std::vector<topology::Cell> leaf_elements;
            std::function<topology::Cell( const topology::Cell& )> to_space_cell;
            const topology::HierarchicalMultiPatchCombinatorialMap* hierarchy = nullptr;
            const param::MultiPatchParametricAtlas* atlas = nullptr;
        };

        // The level, patch and element indices of a level element, whichever dart names it.
        using LevelElementKey = std::vector<size_t>;

        LevelElementKey levelElementKey( const topology::MultiPatchCombinatorialMap& cmap,
                                         const size_t level,
                                         const topology::Cell& element )
        {
            const auto [patch_id, local_d] = cmap.toLocalDart( element.dart() );
            LevelElementKey out{ level, patch_id };
            for( const topology::Dart& d :
                 topology::unflattenFull( *cmap.constituents().at( patch_id ), local_d ).unflat_darts )
                out.push_back( d.id() );
            return out;
        }

        void checkTopDimensionalSpace( const SplineSpace& ss )
        {
            const uint dim = ss.basisComplex().parametricAtlas().cmap().dim();
            if( dim < 2 or dim > 3 )
                throw std::runtime_error( "Trace mesh enumeration is implemented for 2D and 3D spaces." );
        }

        std::vector<LevelTraceMesh> levelTraceMeshes( const SplineSpace& ss )
        {
            checkTopDimensionalSpace( ss );

            const auto* mp_cmap =
                dynamic_cast<const topology::MultiPatchCombinatorialMap*>(
                    &ss.basisComplex().parametricAtlas().cmap() );
            if( mp_cmap != nullptr )
            {
                std::vector<topology::Cell> cells;
                topology::iterateCellsWhile( *mp_cmap, mp_cmap->dim(), [&]( const topology::Cell& c ) {
                    cells.push_back( c );
                    return true;
                } );

                return { LevelTraceMesh{
                    0, mp_cmap, cells, []( const topology::Cell& c ) { return c; }, nullptr, nullptr } };
            }

            const auto* hmp_cmap =
                dynamic_cast<const topology::HierarchicalMultiPatchCombinatorialMap*>(
                    &ss.basisComplex().parametricAtlas().cmap() );
            if( hmp_cmap == nullptr )
                throw std::runtime_error( "Trace mesh enumeration requires a multipatch or hierarchical multipatch space." );

            const auto* hmp_atlas = dynamic_cast<const param::HierarchicalMultiPatchParametricAtlas*>(
                &ss.basisComplex().parametricAtlas() );
            const std::vector<std::vector<topology::Cell>> leaves = topology::leafElements( *hmp_cmap );

            // A leaf's own level darts on a side split by finer neighbours are not
            // leaf darts; such a leaf is named by one of its leaf darts instead.
            const auto leaf_darts = std::make_shared<std::vector<bool>>( hmp_cmap->maxDartId() + 1, false );
            hmp_cmap->iterateDartsWhile( [&]( const topology::Dart& d ) {
                leaf_darts->at( d.id() ) = true;
                return true;
            } );
            const auto leaf_dart_of = std::make_shared<std::map<LevelElementKey, topology::Dart>>();
            hmp_cmap->iterateCellsWhile( hmp_cmap->dim(), [&]( const topology::Cell& c ) {
                const auto [level, level_d] = hmp_cmap->unrefinedAncestorDartOfCell( c );
                const LevelElementKey key = levelElementKey(
                    *hmp_cmap->refinementLevels().at( level ), level, topology::Cell( level_d, c.dim() ) );
                if( not leaf_dart_of->emplace( key, c.dart() ).second )
                    throw std::runtime_error( "Two leaf cells name the same trace mesh element." );
                return true;
            } );

            std::vector<LevelTraceMesh> out;
            out.reserve( hmp_cmap->numLevels() );
            for( size_t level = 0; level < hmp_cmap->numLevels(); level++ )
            {
                out.push_back( LevelTraceMesh{
                    level,
                    hmp_cmap->refinementLevels().at( level ).get(),
                    leaves.at( level ),
                    [hmp_cmap, level, leaf_darts, leaf_dart_of]( const topology::Cell& c ) {
                        const topology::Dart d = hmp_cmap->toGlobalDart( level, c.dart() );
                        if( leaf_darts->at( d.id() ) ) return topology::Cell( d, c.dim() );
                        const auto it = leaf_dart_of->find(
                            levelElementKey( *hmp_cmap->refinementLevels().at( level ), level, c ) );
                        if( it == leaf_dart_of->end() ) throw std::runtime_error( "Trace mesh element is not a leaf." );
                        return topology::Cell( it->second, c.dim() );
                    },
                    hmp_cmap,
                    hmp_atlas == nullptr ? nullptr : hmp_atlas->refinementLevels().at( level ).get() } );
            }
            return out;
        }

        std::vector<size_t> sideLengths( const topology::TPCombinatorialMap& cmap, const ElementSide& side )
        {
            const SmallVector<std::shared_ptr<const topology::CombinatorialMap1d>, 3> components =
                topology::tensorProductComponentCMaps( cmap );

            std::vector<size_t> out;
            for( size_t axis = 0; axis < cmap.dim(); axis++ )
            {
                if( axis == side.axis ) continue;
                const auto& component = components.at( axis );
                out.push_back( topology::cellCount( *component, component->dim() ) );
            }
            return out;
        }

        size_t normalLength( const topology::TPCombinatorialMap& cmap, const ElementSide& side )
        {
            const SmallVector<std::shared_ptr<const topology::CombinatorialMap1d>, 3> components =
                topology::tensorProductComponentCMaps( cmap );
            const auto& component = components.at( side.axis );
            return topology::cellCount( *component, component->dim() );
        }

        topology::Cell localCellOnSide( const topology::TPCombinatorialMap& cmap,
                                        const ElementSide& side,
                                        const SideIndex& side_index )
        {
            topology::FullyUnflattenedDart unflat;
            const size_t n_length = normalLength( cmap, side );
            for( size_t axis = 0, side_axis = 0; axis < cmap.dim(); axis++ )
            {
                if( axis == side.axis )
                    unflat.unflat_darts.push_back( topology::Dart( side.lower ? 0 : n_length - 1 ) );
                else
                    unflat.unflat_darts.push_back( topology::Dart( side_index.at( side_axis++ ) ) );
            }
            unflat.dart_pos =
                SmallVector<topology::TPCombinatorialMap::TPDartPos, 2>(
                    cmap.dim() - 1, topology::TPCombinatorialMap::TPDartPos::DartPos0 );
            return topology::Cell( topology::flattenFull( cmap, unflat ), cmap.dim() );
        }

        topology::Cell levelCellOnSide( const topology::MultiPatchCombinatorialMap& cmap,
                                        const size_t patch_id,
                                        const ElementSide& side,
                                        const SideIndex& side_index )
        {
            const topology::Cell local_cell = localCellOnSide( *cmap.constituents().at( patch_id ), side, side_index );
            return topology::Cell( cmap.toGlobalDart( patch_id, local_cell.dart() ), cmap.dim() );
        }

        SideIndex sideIndex( const topology::TPCombinatorialMap& cmap,
                             const topology::Cell& local_cell,
                             const ElementSide& side )
        {
            const topology::FullyUnflattenedDart unflat = topology::unflattenFull( cmap, local_cell.dart() );
            SideIndex out;
            for( size_t axis = 0; axis < cmap.dim(); axis++ )
            {
                if( axis != side.axis ) out.push_back( unflat.unflat_darts.at( axis ).id() );
            }
            return out;
        }

        std::set<SideIndex> sideLeafIndices( const topology::MultiPatchCombinatorialMap& cmap,
                                             const std::vector<topology::Cell>& leaf_elements,
                                             const size_t patch_id,
                                             const ElementSide& side )
        {
            const auto& patch_cmap = *cmap.constituents().at( patch_id );
            const size_t n_length = normalLength( patch_cmap, side );

            std::set<SideIndex> out;
            for( const topology::Cell& leaf : leaf_elements )
            {
                if( leaf.dim() != cmap.dim() ) continue;
                const auto [leaf_patch, local_d] = cmap.toLocalDart( leaf.dart() );
                if( leaf_patch != patch_id ) continue;

                const topology::FullyUnflattenedDart unflat = topology::unflattenFull( patch_cmap, local_d );
                const size_t normal_index = unflat.unflat_darts.at( side.axis ).id();
                const bool on_side = side.lower ? normal_index == 0 : normal_index + 1 == n_length;
                if( not on_side ) continue;

                out.insert( sideIndex( patch_cmap, topology::Cell( local_d, leaf.dim() ), side ) );
            }
            return out;
        }

        std::vector<SideIndex> allSideIndices( const std::vector<size_t>& lengths )
        {
            std::vector<SideIndex> out{ SideIndex{} };
            for( const size_t n : lengths )
            {
                std::vector<SideIndex> next;
                for( const SideIndex& prefix : out )
                    for( size_t i = 0; i < n; i++ )
                    {
                        next.push_back( prefix );
                        next.back().push_back( i );
                    }
                out = next;
            }
            return out;
        }

        // The children, on the same side, of a level cell on a patch side.
        std::vector<SideIndex> sideChildIndices( const LevelTraceMesh& coarse,
                                                 const LevelTraceMesh& fine,
                                                 const size_t patch_id,
                                                 const ElementSide& side,
                                                 const SideIndex& index )
        {
            if( coarse.hierarchy == nullptr )
                throw std::runtime_error( "Trace mesh refinement levels have no hierarchy." );
            const auto& fine_patch = *fine.cmap->constituents().at( patch_id );
            const size_t n_length = normalLength( fine_patch, side );

            std::vector<SideIndex> out;
            coarse.hierarchy->iterateChildren(
                levelCellOnSide( *coarse.cmap, patch_id, side, index ),
                coarse.level,
                [&]( const topology::Cell& child ) {
                    const topology::Dart local_d = fine.cmap->toLocalDart( child.dart() ).second;
                    const size_t normal_index =
                        topology::unflattenFull( fine_patch, local_d ).unflat_darts.at( side.axis ).id();
                    if( side.lower ? normal_index == 0 : normal_index + 1 == n_length )
                        out.push_back( sideIndex( fine_patch, topology::Cell( local_d, child.dim() ), side ) );
                    return true;
                } );
            std::sort( out.begin(), out.end() );
            return out;
        }

        struct SideLeaf
        {
            size_t level;
            SideIndex index;
        };

        // Where the level-`level` cell at `fragment` on a patch side lies in the side of `leaf`.
        TraceSideEmbedding fragmentEmbedding( const std::vector<LevelTraceMesh>& levels,
                                              const size_t patch_id,
                                              const ElementSide& side,
                                              const size_t level,
                                              const SideIndex& fragment,
                                              const SideLeaf& leaf,
                                              const topology::SideCoordinateTransform& axes )
        {
            const size_t dim = levels.front().cmap->dim();
            TraceSideEmbedding out = wholeTraceSideEmbedding( dim - 1 );
            out.axes = axes;
            if( leaf.level == level ) return out;

            const LevelTraceMesh& fine = levels.at( level );
            const LevelTraceMesh& coarse = levels.at( leaf.level );
            for( const LevelTraceMesh* l : { &fine, &coarse } )
                if( l->atlas == nullptr or &l->atlas->cmap() != l->cmap )
                    throw std::runtime_error( "Trace mesh refinement levels have no matching parametric atlas." );
            const topology::Cell fine_cell = levelCellOnSide( *fine.cmap, patch_id, side, fragment );
            const topology::Cell coarse_cell = levelCellOnSide( *coarse.cmap, patch_id, side, leaf.index );
            const Vector6dMax fine_start = fine.atlas->parametricStarts( fine_cell );
            const Vector6dMax fine_length = fine.atlas->parametricLengths( fine_cell );
            const Vector6dMax coarse_start = coarse.atlas->parametricStarts( coarse_cell );
            const Vector6dMax coarse_length = coarse.atlas->parametricLengths( coarse_cell );
            const std::vector<size_t> tangential = topology::tangentialAxes( dim, side.sideId() );
            for( size_t d = 0; d < tangential.size(); d++ )
            {
                const size_t a = tangential.at( d );
                out.offset[d] = ( fine_start( a ) - coarse_start( a ) ) / coarse_length( a );
                out.scale[d] = fine_length( a ) / coarse_length( a );
            }
            return out;
        }

        std::set<LevelElementKey> levelLeafKeys( const LevelTraceMesh& level )
        {
            std::set<LevelElementKey> out;
            for( const topology::Cell& leaf : level.leaf_elements )
                out.insert( levelElementKey( *level.cmap, level.level, leaf ) );
            return out;
        }

        bool traceHasBrokenRows( const SplineSpace& ss,
                                 const TraceMeshSide& first,
                                 const TraceMeshSide& second,
                                 const double row_tol )
        {
            const TraceSideData first_trace = traceSideData( ss, first.element, first.side, row_tol );
            const TraceSideData second_trace = traceSideData( ss, second.element, second.side, row_tol );

            std::set<FunctionId> first_ids( first_trace.connectivity.begin(), first_trace.connectivity.end() );
            std::set<FunctionId> second_ids( second_trace.connectivity.begin(), second_trace.connectivity.end() );
            std::vector<FunctionId> shared;
            std::set_intersection( first_ids.begin(),
                                   first_ids.end(),
                                   second_ids.begin(),
                                   second_ids.end(),
                                   std::back_inserter( shared ) );
            return shared.size() < std::min( first_ids.size(), second_ids.size() );
        }
    }

    std::vector<size_t> permuteTraceSideIndex( const topology::SideCoordinateTransform& transform,
                                               const std::vector<size_t>& index,
                                               const std::vector<size_t>& lengths )
    {
        return topology::transformSideIndex( transform, index, lengths );
    }

    Eigen::VectorXd permuteTraceSidePoint( const topology::SideCoordinateTransform& transform,
                                           const Eigen::VectorXd& point )
    {
        const std::vector<double> in( point.data(), point.data() + point.size() );
        const std::vector<double> out = topology::transformSidePoint( transform, in );
        return Eigen::Map<const Eigen::VectorXd>( out.data(), static_cast<Eigen::Index>( out.size() ) );
    }

    TraceSideEmbedding wholeTraceSideEmbedding( const size_t rank )
    {
        TraceSideEmbedding out;
        for( size_t d = 0; d < rank; d++ )
        {
            out.axes.source_axis_for_destination.push_back( d );
            out.axes.source_axis_reversed.push_back( false );
            out.offset.push_back( 0.0 );
            out.scale.push_back( 1.0 );
        }
        validateTraceSideEmbedding( out );
        return out;
    }

    void validateTraceSideEmbedding( const TraceSideEmbedding& embedding, const double tol )
    {
        topology::validateSideCoordinateTransform( embedding.axes );
        const size_t rank = embedding.axes.source_axis_for_destination.size();
        if( embedding.offset.size() != rank or embedding.scale.size() != rank )
            throw std::invalid_argument( "Trace side embedding offset and scale must match the axis rank." );
        for( size_t d = 0; d < rank; d++ )
        {
            const double offset = embedding.offset.at( d );
            const double scale = embedding.scale.at( d );
            if( not std::isfinite( offset ) or not std::isfinite( scale ) )
                throw std::invalid_argument( "Trace side embedding offset and scale must be finite." );
            if( scale <= 0.0 ) throw std::invalid_argument( "Trace side embedding scales must be positive." );
            if( offset < -tol or offset + scale > 1.0 + tol )
                throw std::invalid_argument( "Trace side embedding places the fragment outside its leaf side." );
        }
    }

    Eigen::VectorXd embedTraceFragmentPoint( const TraceSideEmbedding& embedding, const Eigen::VectorXd& xi )
    {
        validateTraceSideEmbedding( embedding );
        Eigen::VectorXd y = permuteTraceSidePoint( embedding.axes, xi );
        for( Eigen::Index d = 0; d < y.size(); d++ )
            y( d ) = embedding.offset.at( d ) + embedding.scale.at( d ) * y( d );
        return y;
    }

    void validateTraceMeshInterface( const TraceMeshInterface& iface, const double tol )
    {
        validateTraceSideEmbedding( iface.first.embedding, tol );
        const size_t rank = iface.first.embedding.offset.size();
        if( iface.second.has_value() )
        {
            validateTraceSideEmbedding( iface.second->embedding, tol );
            if( iface.second->embedding.offset.size() != rank )
                throw std::invalid_argument( "Trace record sides have embeddings of different rank." );
        }

        const auto identity_axes = [rank]( const TraceSideEmbedding& e ) {
            for( size_t d = 0; d < rank; d++ )
                if( e.axes.source_axis_for_destination.at( d ) != d or e.axes.source_axis_reversed.at( d ) )
                    return false;
            return true;
        };
        const auto whole_side = [&]( const TraceSideEmbedding& e ) {
            if( not identity_axes( e ) ) return false;
            for( size_t d = 0; d < rank; d++ )
                if( std::abs( e.offset.at( d ) ) > tol or std::abs( e.scale.at( d ) - 1.0 ) > tol ) return false;
            return true;
        };

        if( iface.type == TraceMeshEntityType::Boundary and
            ( iface.second.has_value() or iface.permutation.has_value() ) )
            throw std::invalid_argument( "A boundary trace record must have no second side and no permutation." );
        if( iface.type == TraceMeshEntityType::Interior and
            ( not iface.second.has_value() or iface.permutation.has_value() ) )
            throw std::invalid_argument( "An interior trace record must have a second side and no permutation." );
        if( iface.type != TraceMeshEntityType::PatchInterface )
        {
            if( not whole_side( iface.first.embedding ) or
                ( iface.second.has_value() and not whole_side( iface.second->embedding ) ) )
                throw std::invalid_argument(
                    "Boundary and interior trace records must embed whole sides by the identity." );
            return;
        }

        if( not iface.second.has_value() or not iface.permutation.has_value() )
            throw std::invalid_argument( "A patch interface trace record needs a second side and a permutation." );
        const topology::SideCoordinateTransform transform = topology::sideCoordinateTransform(
            rank + 1, iface.first.side.sideId(), iface.second->side.sideId(), iface.permutation.value() );
        bool frame = identity_axes( iface.first.embedding );
        for( size_t d = 0; d < rank; d++ )
            frame = frame and
                    iface.second->embedding.axes.source_axis_for_destination.at( d ) ==
                        transform.source_axis_for_destination.at( d ) and
                    iface.second->embedding.axes.source_axis_reversed.at( d ) == transform.source_axis_reversed.at( d );
        if( not frame )
            throw std::invalid_argument(
                "A patch interface fragment must use the first side's axes and the interface transform." );
    }


    std::vector<TraceMeshInterface> boundaryTraceMeshInterfaces( const SplineSpace& ss )
    {
        std::vector<TraceMeshInterface> out;
        for( const LevelTraceMesh& level : levelTraceMeshes( ss ) )
        {
            for( size_t patch_id = 0; patch_id < level.cmap->constituents().size(); patch_id++ )
            {
                for( size_t side_id = 0; side_id < 2 * level.cmap->dim(); side_id++ )
                {
                    if( level.cmap->connections().contains( { patch_id, side_id } ) ) continue;

                    const ElementSide side = elementSideFromId( side_id );
                    for( const SideIndex& index : sideLeafIndices( *level.cmap, level.leaf_elements, patch_id, side ) )
                    {
                        const topology::Cell level_cell = levelCellOnSide( *level.cmap, patch_id, side, index );
                        out.push_back( { TraceMeshEntityType::Boundary,
                                         TraceMeshSide{ level.to_space_cell( level_cell ),
                                                        side,
                                                        patch_id,
                                                        level.level,
                                                        wholeTraceSideEmbedding( level.cmap->dim() - 1 ) },
                                         std::nullopt,
                                         std::nullopt } );
                        validateTraceMeshInterface( out.back() );
                    }
                }
            }
        }
        return out;
    }

    std::vector<TraceMeshInterface> patchTraceMeshInterfaces( const SplineSpace& ss )
    {
        const std::vector<LevelTraceMesh> levels = levelTraceMeshes( ss );
        const size_t dim = levels.front().cmap->dim();
        const topology::SideCoordinateTransform identity = wholeTraceSideEmbedding( dim - 1 ).axes;

        std::vector<TraceMeshInterface> out;
        for( const auto& [first, connection] : levels.front().cmap->connections() )
        {
            const topology::MultiPatchCombinatorialMap::ConstituentSide& second = connection.second;
            if( not( first < second ) ) continue;

            const ElementSide first_side = elementSideFromId( first.side_id );
            const ElementSide second_side = elementSideFromId( second.side_id );
            const topology::SideCoordinateTransform transform =
                topology::sideCoordinateTransform( dim, first.side_id, second.side_id, connection.first );

            std::vector<std::vector<size_t>> first_lengths;
            std::vector<std::set<SideIndex>> first_leaves;
            std::vector<std::set<SideIndex>> second_leaves;
            for( const LevelTraceMesh& level : levels )
            {
                const auto& first_cmap = *level.cmap->constituents().at( first.constituent_id );
                const auto& second_cmap = *level.cmap->constituents().at( second.constituent_id );
                first_lengths.push_back( sideLengths( first_cmap, first_side ) );
                if( topology::transformedExtents( transform, first_lengths.back() ) !=
                    sideLengths( second_cmap, second_side ) )
                    throw std::invalid_argument( "Patch interface sides have mismatched element counts." );
                first_leaves.push_back(
                    sideLeafIndices( *level.cmap, level.leaf_elements, first.constituent_id, first_side ) );
                second_leaves.push_back(
                    sideLeafIndices( *level.cmap, level.leaf_elements, second.constituent_id, second_side ) );
            }

            const auto leaf_side = [&]( const topology::MultiPatchCombinatorialMap::ConstituentSide& cs,
                                        const ElementSide& side,
                                        const SideLeaf& leaf,
                                        const size_t level,
                                        const SideIndex& fragment,
                                        const topology::SideCoordinateTransform& axes ) {
                const LevelTraceMesh& leaf_level = levels.at( leaf.level );
                return TraceMeshSide{
                    leaf_level.to_space_cell(
                        levelCellOnSide( *leaf_level.cmap, cs.constituent_id, side, leaf.index ) ),
                    side,
                    cs.constituent_id,
                    leaf.level,
                    fragmentEmbedding( levels, cs.constituent_id, side, level, fragment, leaf, axes ) };
            };

            // Descend from each level-0 interface cell, keeping the first leaf met on
            // each side; the cell where both sides have one is a common fragment.
            std::function<void( size_t, const SideIndex&, std::optional<SideLeaf>, std::optional<SideLeaf> )> descend =
                [&]( const size_t level,
                     const SideIndex& index,
                     std::optional<SideLeaf> first_leaf,
                     std::optional<SideLeaf> second_leaf ) {
                    const SideIndex second_index = permuteTraceSideIndex( transform, index, first_lengths.at( level ) );
                    if( not first_leaf and first_leaves.at( level ).contains( index ) )
                        first_leaf = SideLeaf{ level, index };
                    if( not second_leaf and second_leaves.at( level ).contains( second_index ) )
                        second_leaf = SideLeaf{ level, second_index };

                    if( first_leaf and second_leaf )
                    {
                        out.push_back( { TraceMeshEntityType::PatchInterface,
                                         leaf_side( first, first_side, *first_leaf, level, index, identity ),
                                         leaf_side( second, second_side, *second_leaf, level, second_index, transform ),
                                         connection.first } );
                        validateTraceMeshInterface( out.back() );
                        return;
                    }
                    if( level + 1 == levels.size() )
                        throw std::runtime_error( "A patch interface cell has no leaf on one of its sides." );
                    for( const SideIndex& child : sideChildIndices(
                             levels.at( level ), levels.at( level + 1 ), first.constituent_id, first_side, index ) )
                        descend( level + 1, child, first_leaf, second_leaf );
                };
            for( const SideIndex& index : allSideIndices( first_lengths.front() ) )
                descend( 0, index, std::nullopt, std::nullopt );
        }
        return out;
    }

    std::vector<TraceMeshInterface> interiorTraceMeshInterfaces( const SplineSpace& ss,
                                                                 const InteriorTraceMode mode,
                                                                 const double row_tol )
    {
        std::vector<TraceMeshInterface> out;
        for( const LevelTraceMesh& level : levelTraceMeshes( ss ) )
        {
            const std::set<LevelElementKey> leaves = levelLeafKeys( level );
            std::set<std::pair<LevelElementKey, LevelElementKey>> seen_pairs;

            for( const topology::Cell& level_leaf : level.leaf_elements )
            {
                const auto [patch_id, local_d] = level.cmap->toLocalDart( level_leaf.dart() );
                const auto& patch_cmap = *level.cmap->constituents().at( patch_id );
                const topology::Cell local_leaf( local_d, level_leaf.dim() );
                const LevelElementKey leaf_key = levelElementKey( *level.cmap, level.level, level_leaf );

                for( size_t side_id = 0; side_id < 2 * level.cmap->dim(); side_id++ )
                {
                    const ElementSide side = elementSideFromId( side_id );
                    const auto adjacent = adjacentElementSide( patch_cmap, local_leaf, side );
                    if( not adjacent.has_value() ) continue;
                    const topology::Cell adjacent_level_cell(
                        level.cmap->toGlobalDart( patch_id, adjacent->first.dart() ), level.cmap->dim() );
                    const LevelElementKey adjacent_key =
                        levelElementKey( *level.cmap, level.level, adjacent_level_cell );
                    if( not leaves.contains( adjacent_key ) ) continue;
                    if( not seen_pairs.insert( std::minmax( leaf_key, adjacent_key ) ).second ) continue;

                    const TraceSideEmbedding whole = wholeTraceSideEmbedding( level.cmap->dim() - 1 );
                    const TraceMeshSide leaf_side{
                        level.to_space_cell( level_leaf ), side, patch_id, level.level, whole };
                    const TraceMeshSide adjacent_side{
                        level.to_space_cell( adjacent_level_cell ), adjacent->second, patch_id, level.level, whole };
                    TraceMeshInterface iface{ TraceMeshEntityType::Interior,
                                              side.lower ? adjacent_side : leaf_side,
                                              side.lower ? leaf_side : adjacent_side,
                                              std::nullopt };
                    validateTraceMeshInterface( iface );

                    if( mode == InteriorTraceMode::Broken and
                        not traceHasBrokenRows( ss, iface.first, iface.second.value(), row_tol ) )
                        continue;

                    out.push_back( iface );
                }
            }
        }
        return out;
    }

    std::vector<TraceMeshInterface> traceMeshInterfaces( const SplineSpace& ss,
                                                         const bool include_boundary,
                                                         const bool include_patch_interfaces,
                                                         const bool include_interior,
                                                         const InteriorTraceMode interior_mode,
                                                         const double row_tol )
    {
        std::vector<TraceMeshInterface> out;
        const auto append = [&out]( const std::vector<TraceMeshInterface>& more ) {
            out.insert( out.end(), more.begin(), more.end() );
        };

        if( include_boundary ) append( boundaryTraceMeshInterfaces( ss ) );
        if( include_patch_interfaces ) append( patchTraceMeshInterfaces( ss ) );
        if( include_interior ) append( interiorTraceMeshInterfaces( ss, interior_mode, row_tol ) );
        return out;
    }

    TraceInterfaceElement traceInterfaceElement( const SplineSpace& ss,
                                                 const TraceMeshInterface& iface,
                                                 const double row_tol )
    {
        TraceInterfaceElement out{ traceSideData( ss, iface.first.element, iface.first.side, row_tol ), std::nullopt };
        if( iface.second.has_value() )
        {
            out.second = traceSideData( ss, iface.second->element, iface.second->side, row_tol );
        }
        return out;
    }
}
