#include <MultiPatchCombinatorialMap.hpp>
#include <IndexOperations.hpp>
#include <SideCoordinateTransform.hpp>
#include <TPCombinatorialMap.hpp>
#include <utility>
#include <CombinatorialMapMethods.hpp>
#include <GlobalCellMarker.hpp>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <Logging.hpp>

namespace topology
{
    using TPPermutation = MultiPatchCombinatorialMap::TPPermutation;
    using TPDartPos = TPCombinatorialMap::TPDartPos;

    static constexpr std::array<std::array<Dart::IndexType, 4>, 6> local_darts_from_bdry_ids_3d{
        std::array<Dart::IndexType, 4>{ 7, 8, 9, 10 },
        std::array<Dart::IndexType, 4>{ 19, 20, 21, 22 },
        std::array<Dart::IndexType, 4>{ 13, 14, 15, 16 },
        std::array<Dart::IndexType, 4>{ 1, 2, 3, 4 },
        std::array<Dart::IndexType, 4>{ 5, 23, 17, 11 },
        std::array<Dart::IndexType, 4>{ 0, 6, 12, 18 } };

    // Boundaries are specified by the entry in BoundaryBoolVec that would be true.
    // This results in the ordering S1, S0, T1, T0, ...
    static constexpr std::array<size_t, 4> local_dart_to_bdry_2d{ 3, 0, 2, 1 };
    static constexpr std::array<Dart::IndexType, 4> local_dart_from_bdry_2d{ 1, 3, 2, 0 };

    Dart flipDart( const util::IndexVec& tp_lengths, const SmallVector<Dart, 2> from_darts, const size_t id )
    {
        return Dart( tp_lengths.at( id ) - from_darts.at( id ).id() - 1 );
    }

    SmallVector<Dart, 2> flipNecessaryDarts( const util::IndexVec& bdry_lengths,
                                             const SmallVector<Dart, 2>& bdry_unflat,
                                             const size_t which_boundary )
    {
        if( bdry_unflat.size() == 1 )
        {
            if( which_boundary == 1 or which_boundary == 2 )
                return { flipDart( bdry_lengths, bdry_unflat, 0 ) };
            else
                return bdry_unflat;
        }
        else
        {
            switch( which_boundary )
            {
                case 0:
                case 3:
                case 4: return { flipDart( bdry_lengths, bdry_unflat, 0 ), bdry_unflat.at( 1 ) };
                case 1:
                case 2:
                case 5: return bdry_unflat;
                default: throw std::runtime_error( "Bad boundary" );
            }
        }
    };

    std::pair<SmallVector<Dart, 2>, TPDartPos> permuteAcrossBoundary( const SmallVector<Dart, 2>& from_darts,
                                                                      const TPDartPos& from_pos,
                                                                      const TPPermutation& permute,
                                                                      const util::IndexVec& tp_lengths )
    {
        if( tp_lengths.size() != from_darts.size() )
            throw std::runtime_error( "tp_lengths size must match from_darts size" );

        if( from_darts.size() != ( permute == TPPermutation::Flip1d ? 1 : 2 ) )
            throw std::runtime_error( "Bad inputs to permuteAcrossBoundary." );

        const TPDartPos permuted_pos =
            TPDartPos( ( std::to_underlying( permute ) + ( 4 - std::to_underlying( from_pos ) ) ) % 4 );

        switch( permute )
        {
            case TPPermutation::Flip1d:
                // 1d boundary, just flip the dart id
                return { { flipDart( tp_lengths, from_darts, 0 ) }, from_pos };
            case TPPermutation::ZeroToZero:
                return { { flipDart( tp_lengths, from_darts, 0 ), from_darts.at( 1 ) }, permuted_pos };
            case TPPermutation::ZeroToOne:
                return { { flipDart( tp_lengths, from_darts, 1 ), flipDart( tp_lengths, from_darts, 0 ) },
                         permuted_pos };
            case TPPermutation::ZeroToTwo:
                return { { from_darts.at( 0 ), flipDart( tp_lengths, from_darts, 1 ) }, permuted_pos };
            case TPPermutation::ZeroToThree: return { { from_darts.at( 1 ), from_darts.at( 0 ) }, permuted_pos };
            default: return { {}, TPDartPos::DartPos0 };// Unreachable; to silence warning on ubuntu
        }
    }

    size_t whichBoundary( const TPCombinatorialMap& cmap, const Dart& d )
    {
        if( cmap.dim() == 3 )
        {
            const Dart::IndexType hex_local_d = d.id() % 24;
            for( size_t i = 0; i < local_darts_from_bdry_ids_3d.size(); i++ )
            {
                if( std::ranges::find( local_darts_from_bdry_ids_3d.at( i ), hex_local_d ) !=
                    local_darts_from_bdry_ids_3d.at( i ).end() )
                {
                    return i;
                }
            }
            throw std::runtime_error( "Dart on no boundary!" );
        }
        else
            return local_dart_to_bdry_2d.at( d.id() % 4 );
    }

    std::pair<SmallVector<Dart, 2>, TPDartPos> dropToBoundary( const TPCombinatorialMap& cmap,
                                                               const Dart& d,
                                                               const size_t which_boundary,
                                                               const util::IndexVec& bdry_lengths )
    {
        const TPDartPos dart_pos = [&]() {
            if( cmap.dim() == 3 )
            {
                const auto& local_darts_from_bdry = local_darts_from_bdry_ids_3d.at( which_boundary );
                const auto it = std::find( local_darts_from_bdry.begin(), local_darts_from_bdry.end(), d.id() % 24 );
                if( it == local_darts_from_bdry.end() ) throw std::runtime_error( "Dart not in provided boundary!" );
                return TPDartPos( std::distance( local_darts_from_bdry.begin(), it ) );
            }
            else
                return TPDartPos( local_dart_to_bdry_2d.at( d.id() % 4 ) );
        }();

        const SmallVector<Dart, 3> full_unflat = unflattenFull( cmap, d ).unflat_darts;

        // Drop the correct dart to get to the boundary
        SmallVector<Dart, 2> bdry_unflat;
        for( size_t i = 0; i < full_unflat.size(); i++ )
        {
            if( i != which_boundary / 2 ) bdry_unflat.push_back( full_unflat.at( i ) );
        }

        return { flipNecessaryDarts( bdry_lengths, bdry_unflat, which_boundary ), dart_pos };
    }

    util::IndexVec boundaryLengths( const TPCombinatorialMap& cmap, const size_t which_boundary )
    {
        const auto comps = tensorProductComponentCMaps( cmap );

        util::IndexVec out;
        for( size_t i = 0; i < comps.size(); i++ )
        {
            if( i == which_boundary / 2 ) continue;
            const auto& comp = comps.at( i );
            out.push_back( cellCount( *comp, comp->dim() ) );
        }

        return out;
    }

    size_t perpLength( const TPCombinatorialMap& cmap, const size_t which_boundary )
    {
        const auto comps = tensorProductComponentCMaps( cmap );

        for( size_t i = 0; i < comps.size(); i++ )
        {
            const auto& comp = comps.at( i );
            if( i == which_boundary / 2 ) return cellCount( *comp, comp->dim() );
        }

        throw std::runtime_error( "Bad boundary specified" );
    }

    Dart raiseFromBoundary( const TPCombinatorialMap& cmap,
                            const SmallVector<Dart, 2>& from_darts,
                            const TPDartPos& from_pos,
                            const size_t which_boundary,
                            const util::IndexVec& bdry_lengths )
    {
        const auto bdry_darts = flipNecessaryDarts( bdry_lengths, from_darts, which_boundary );
        SmallVector<Dart, 3> unflat_darts;
        for( size_t i = 0; i <= bdry_darts.size(); i++ )
        {
            if( i == which_boundary / 2 )
            {
                unflat_darts.push_back( [&]() {
                    if( which_boundary % 2 == 1 )
                        return Dart( 0 );
                    else
                        return Dart( perpLength( cmap, which_boundary ) - 1 );
                }() );
            }
            if( i < bdry_darts.size() ) unflat_darts.push_back( bdry_darts.at( i ) );
        }

        const Dart::IndexType local_dart_id =
            cmap.dim() == 2 ? local_dart_from_bdry_2d.at( which_boundary )
                            : local_darts_from_bdry_ids_3d.at( which_boundary ).at( std::to_underlying( from_pos ) );

        return Dart( flattenFull( cmap, unflat_darts ).id() + local_dart_id );
    }

    namespace
    {
        // Run first by both public constructors, before anything dereferences a
        // patch, so the two admit the same constituent lists.  Dimension is
        // checked across the whole list rather than per connection: dim()
        // reports only the first patch's, so differing patches that are never
        // glued to each other would otherwise pass.
        void validateConstituents( const std::vector<std::shared_ptr<const TPCombinatorialMap>>& constituents )
        {
            if( constituents.empty() ) throw std::runtime_error( "Cannot create multipatch of zero patches" );

            for( size_t i = 0; i < constituents.size(); i++ )
            {
                std::ostringstream msg;
                if( constituents.at( i ) == nullptr )
                {
                    msg << "Multipatch constituent " << i << " is a null patch.";
                    throw std::runtime_error( msg.str() );
                }
                if( constituents.at( i )->dim() != constituents.front()->dim() )
                {
                    msg << "Multipatch constituent " << i << " has dimension " << constituents.at( i )->dim()
                        << ", but constituent 0 has dimension " << constituents.front()->dim()
                        << "; all patches must have the same dimension.";
                    throw std::runtime_error( msg.str() );
                }
            }
        }
    } // namespace

    using ConstituentSide = MultiPatchCombinatorialMap::ConstituentSide;
    std::map<ConstituentSide, std::pair<TPPermutation, ConstituentSide>> initializeInterMapConnections(
        const std::vector<std::shared_ptr<const TPCombinatorialMap>>& constituents,
        const std::map<std::pair<size_t, Dart>, std::pair<size_t, Dart>>& connections )
    {
        validateConstituents( constituents );

        std::map<ConstituentSide, std::pair<TPPermutation, ConstituentSide>> out;

        // A patch side has at most one whole-side partner.  Restating the same
        // gluing is harmless; assigning a second partner is rejected rather
        // than silently dropped, as map::emplace would.
        const auto addConnection = [&out]( const ConstituentSide& from,
                                           const TPPermutation perm,
                                           const ConstituentSide& to ) {
            const auto [it, inserted] = out.insert( { from, { perm, to } } );
            if( inserted ) return;

            const auto& [held_perm, held_to] = it->second;
            if( held_perm == perm and held_to.constituent_id == to.constituent_id and
                held_to.side_id == to.side_id )
            {
                return;
            }

            std::ostringstream msg;
            msg << "Patch side " << from << " is already connected to " << held_to << " by " << held_perm
                << ", so it cannot also connect to " << to << " by " << perm
                << ".  A patch side supports at most one whole-side partner.";
            throw std::runtime_error( msg.str() );
        };

        for( const auto& connection : connections )
        {
            const TPCombinatorialMap& left_cmap = *constituents.at( connection.first.first );
            const TPCombinatorialMap& right_cmap = *constituents.at( connection.second.first );
            const size_t left_bdry = whichBoundary( left_cmap, connection.first.second );
            const size_t right_bdry = whichBoundary( right_cmap, connection.second.second );

            const ConstituentSide left_side{ connection.first.first, left_bdry };
            const ConstituentSide right_side{ connection.second.first, right_bdry };

            const TPPermutation perm = [&]() -> TPPermutation {
                if( constituents.front()->dim() == 2 ) return TPPermutation::Flip1d;

                const TPDartPos left_pos =
                    dropToBoundary(
                        left_cmap, connection.first.second, left_bdry, boundaryLengths( left_cmap, left_bdry ) )
                        .second;
                const TPDartPos right_pos =
                    dropToBoundary(
                        right_cmap, connection.second.second, right_bdry, boundaryLengths( right_cmap, right_bdry ) )
                        .second;

                return TPPermutation( ( std::to_underlying( left_pos ) + std::to_underlying( right_pos ) ) % 4 );
            }();

            addConnection( left_side, perm, right_side );
            addConnection( right_side, perm, left_side );
        }

        return out;
    }

    PatchSideInterface interfaceAtDart( const MultiPatchCombinatorialMap& cmap, const Dart& global_dart )
    {
        const auto [patch_id, local_dart] = cmap.toLocalDart( global_dart );
        const TPCombinatorialMap& patch = *cmap.constituents().at( patch_id );

        // whichBoundary reads an ELEMENT-local face direction, which an interior
        // face has just as much as a boundary one, so the patch-boundary test has
        // to come first.  Without it an interior face silently borrows whatever
        // connection is registered on the exterior side of the same direction.
        if( phi( patch, static_cast<int>( patch.dim() ), local_dart ).has_value() )
        {
            std::ostringstream msg;
            msg << "Dart " << global_dart.id() << " lies on an interior face of patch " << patch_id
                << ", which is on no patch interface.";
            throw std::runtime_error( msg.str() );
        }

        const ConstituentSide source{ patch_id, whichBoundary( patch, local_dart ) };

        const auto found = cmap.connections().find( source );
        if( found == cmap.connections().end() )
        {
            std::ostringstream msg;
            msg << "Patch side " << source << " is an unglued exterior side, so it has no interface.";
            throw std::runtime_error( msg.str() );
        }
        const auto& [permutation, destination] = found->second;

        const auto back = cmap.connections().find( destination );
        if( back == cmap.connections().end() or back->second.first != permutation or
            back->second.second.constituent_id != source.constituent_id or
            back->second.second.side_id != source.side_id )
        {
            std::ostringstream msg;
            msg << "Stored connectivity is inconsistent: patch side " << source << " names " << destination
                << " as its partner, which does not name it back by the same permutation.";
            throw std::runtime_error( msg.str() );
        }

        return { source, destination, permutation };
    }

    namespace
    {
        // A map key gives only one partner per SOURCE side, which is weaker than
        // the invariant: it permits A -> B alongside C -> B, a missing reverse
        // entry, and a reverse entry naming a different permutation.  Requiring
        // every connection to be named back identically rules out all three,
        // because the single entry for B can name back only one of A and C.
        const std::vector<std::shared_ptr<const TPCombinatorialMap>>& validateMultiPatchStructure(
            const std::vector<std::shared_ptr<const TPCombinatorialMap>>& constituents,
            const std::map<ConstituentSide, std::pair<TPPermutation, ConstituentSide>>& connections )
        {
            validateConstituents( constituents );

            const auto malformed = [&]( const ConstituentSide& from,
                                        const ConstituentSide& to,
                                        const TPPermutation perm,
                                        const std::string& why ) {
                std::ostringstream msg;
                msg << "Malformed patch connection " << from << " -> " << to << " by " << perm << ": " << why;
                return std::runtime_error( msg.str() );
            };

            for( const auto& [from, connection] : connections )
            {
                const auto& [perm, to] = connection;

                if( from.constituent_id >= constituents.size() or to.constituent_id >= constituents.size() )
                    throw malformed( from, to, perm, "a patch id is outside the constituent list." );

                // Unreachable while the whole-list rule above holds; retained so
                // relaxing that rule cannot silently admit a mixed-dimension gluing.
                const size_t dim = constituents.at( from.constituent_id )->dim();
                if( constituents.at( to.constituent_id )->dim() != dim )
                    throw malformed( from, to, perm, "the two patches have different dimensions." );

                if( from.constituent_id == to.constituent_id and from.side_id == to.side_id )
                    throw malformed( from, to, perm, "a patch side cannot be glued to itself." );

                // The single authority on the dimension, the side ids, and the
                // permutation/dimension pairing.  Its reason is re-reported here
                // so every refusal names the connection it came from.
                try
                {
                    sideCoordinateTransform( dim, from.side_id, to.side_id, perm );
                }
                catch( const std::invalid_argument& e )
                {
                    throw malformed( from, to, perm, e.what() );
                }

                const auto back = connections.find( to );
                if( back == connections.end() )
                    throw malformed( from, to, perm, "the partner side registers no connection of its own." );
                if( back->second.first != perm )
                    throw malformed( from, to, perm, "the partner side names a different permutation." );
                if( back->second.second.constituent_id != from.constituent_id or
                    back->second.second.side_id != from.side_id )
                    throw malformed( from, to, perm, "the partner side names a different side in return." );
            }
            return constituents;
        }
    } // namespace

    MultiPatchCombinatorialMap::MultiPatchCombinatorialMap(
        const std::vector<std::shared_ptr<const TPCombinatorialMap>>& constituents,
        const std::map<std::pair<size_t, Dart>, std::pair<size_t, Dart>>& connections )
        : MultiPatchCombinatorialMap( constituents, initializeInterMapConnections( constituents, connections ) )
    {}

    MultiPatchCombinatorialMap::MultiPatchCombinatorialMap(
        const std::vector<std::shared_ptr<const TPCombinatorialMap>>& constituents,
        const std::map<ConstituentSide, std::pair<TPPermutation, ConstituentSide>>& connections )
        // Validated in the first member's initializer, so before initializeRanges
        // dereferences each patch.
        : mSubMaps( validateMultiPatchStructure( constituents, connections ) ),
          mInterMapConnections( connections ),
          mRanges( initializeRanges( constituents ) )
    {}

    std::pair<size_t, Dart> MultiPatchCombinatorialMap::toLocalDart( const Dart& global_dart ) const
    {
        return mRanges.toLocalDart( global_dart );
    }

    Dart MultiPatchCombinatorialMap::toGlobalDart( const size_t patch_id, const Dart& local_dart ) const
    {
        return mRanges.toGlobalDart( patch_id, local_dart );
    }

    std::optional<Dart> MultiPatchCombinatorialMap::phi( const int i, const Dart& d ) const
    {
        const auto [patch_id, local_d] = toLocalDart( d );

        const TPCombinatorialMap& patch_cmap = *mSubMaps.at( patch_id );

        return topology::phi( patch_cmap, i, local_d )
            .transform( [&]( const Dart& local_phi ) { return toGlobalDart( patch_id, local_phi ); } )
            .or_else( [&]() -> std::optional<Dart> {
                const size_t bdry_id = whichBoundary( patch_cmap, local_d );
                const auto it = mInterMapConnections.find( { patch_id, bdry_id } );
                if( it == mInterMapConnections.end() ) return {};

                const std::pair<TPPermutation, ConstituentSide>& connection = it->second;
                const util::IndexVec bdry_lengths = boundaryLengths( patch_cmap, bdry_id );

                const std::pair<SmallVector<Dart, 2>, TPDartPos> bdry_darts =
                    dropToBoundary( patch_cmap, local_d, bdry_id, bdry_lengths );
                const std::pair<SmallVector<Dart, 2>, TPDartPos> next_bdry_darts =
                    permuteAcrossBoundary( bdry_darts.first, bdry_darts.second, connection.first, bdry_lengths );

                const Dart next_local_d = raiseFromBoundary(
                    *mSubMaps.at( connection.second.constituent_id ),
                    next_bdry_darts.first,
                    next_bdry_darts.second,
                    connection.second.side_id,
                    boundaryLengths( *mSubMaps.at( connection.second.constituent_id ), connection.second.side_id ) );

                return toGlobalDart( connection.second.constituent_id, next_local_d );
            } );
    }

    Dart::IndexType MultiPatchCombinatorialMap::maxDartId() const
    {
        return mRanges.maxDartId();
    }

    uint MultiPatchCombinatorialMap::dim() const
    {
        return mSubMaps.front()->dim();
    }

    bool MultiPatchCombinatorialMap::iterateDartsWhile( const std::function<bool( const Dart& )>& callback ) const
    {
        for( size_t patch_id = 0; patch_id < mSubMaps.size(); patch_id++ )
        {
            const bool continue_iter =
                topology::iterateDartsWhile( *mSubMaps.at( patch_id ), [&]( const Dart& local_dart ) {
                    return callback( toGlobalDart( patch_id, local_dart ) );
                } );
            if( not continue_iter ) return false;
        }
        return true;
    }

    bool MultiPatchCombinatorialMap::iterateCellsWhile( const uint cell_dim,
                                                        const std::function<bool( const Cell& )>& callback ) const
    {
        if( cell_dim == dim() )
        {
            for( size_t patch_id = 0; patch_id < mSubMaps.size(); patch_id++ )
            {
                const bool continue_iter =
                    topology::iterateCellsWhile( *mSubMaps.at( patch_id ), cell_dim, [&]( const Cell& local_cell ) {
                        return callback( Cell( toGlobalDart( patch_id, local_cell.dart() ), local_cell.dim() ) );
                    } );

                if( not continue_iter ) return false;
            }
            return true;
        }
        else
        {
            GlobalCellMarker m( *this, cell_dim );
            for( size_t patch_id = 0; patch_id < mSubMaps.size(); patch_id++ )
            {
                const bool continue_iter =
                    topology::iterateCellsWhile( *mSubMaps.at( patch_id ), cell_dim, [&]( const Cell& local_cell ) {
                        const Cell global_cell( toGlobalDart( patch_id, local_cell.dart() ), local_cell.dim() );
                        if( not m.isMarked( global_cell ) )
                        {
                            m.mark( *this, global_cell );
                            return callback( global_cell );
                        }
                        return true;
                    } );

                if( not continue_iter ) return false;
            }
            return true;
        }
    }

    std::optional<IndexingFunc> MultiPatchCombinatorialMap::indexing( const uint ) const
    {
        return std::nullopt;
        // std::vector<IndexingFunc> indexing_funcs;
        // indexing_funcs.reserve( mSubMaps.size() );
        // for( const auto& sub : mSubMaps )
        // {
        //     const std::optional<IndexingFunc> sub_idx = sub->indexing( cell_dim );
        //     if( not sub_idx.has_value() ) return std::nullopt;
        //     indexing_funcs.push_back( sub_idx.value() );
        // }
        // return [&this, indexing_funcs]( const Cell& c ) {
        //     const Cell local_cell( toLocalDart)
        // }
    }

    std::optional<size_t> MultiPatchCombinatorialMap::cellCount( const uint cell_dim ) const
    {
        if( cell_dim != dim() ) return std::nullopt;

        return std::transform_reduce( mSubMaps.begin(),
                                      mSubMaps.end(),
                                      0,
                                      std::plus<>(),
                                      [&cell_dim]( const std::shared_ptr<const TPCombinatorialMap>& m ) {
                                          return topology::cellCount( *m, cell_dim );
                                      } );
    }

    bool MultiPatchCombinatorialMap::ConstituentSide::operator<(
        const MultiPatchCombinatorialMap::ConstituentSide& o ) const
    {
        if( constituent_id == o.constituent_id )
            return side_id < o.side_id;
        else
            return constituent_id < o.constituent_id;
    }

    std::ostream& operator<<( std::ostream& o, const MultiPatchCombinatorialMap::ConstituentSide& cs )
    {
        o << "{" << cs.constituent_id << ", " << cs.side_id << "}";
        return o;
    }
    std::ostream& operator<<( std::ostream& o, const MultiPatchCombinatorialMap::TPPermutation& tpp )
    {
        switch( tpp )
        {
            case MultiPatchCombinatorialMap::TPPermutation::Flip1d:
                o << "Flip1d";
                break;
            case MultiPatchCombinatorialMap::TPPermutation::ZeroToOne:
                o << "ZeroToOne";
                break;
            case MultiPatchCombinatorialMap::TPPermutation::ZeroToTwo:
                o << "ZeroToTwo";
                break;
            case MultiPatchCombinatorialMap::TPPermutation::ZeroToThree:
                o << "ZeroToThree";
                break;
            case MultiPatchCombinatorialMap::TPPermutation::ZeroToZero:
                o << "ZeroToZero";
                break;
        }
        return o;
    }

    MultiPatchCombinatorialMap::InternalConnectionsMap
        connectionsOfSweptMultipatch( const MultiPatchCombinatorialMap::InternalConnectionsMap& connections_2d )
    {
        using InternalConnectionsMap = MultiPatchCombinatorialMap::InternalConnectionsMap;
        using ConstituentSide = MultiPatchCombinatorialMap::ConstituentSide;
        using TPPermutation = MultiPatchCombinatorialMap::TPPermutation;

        InternalConnectionsMap out;
        for( const auto& [left_side, perm_and_right_side] : connections_2d )
        {
            if( perm_and_right_side.first != TPPermutation::Flip1d )
                throw std::runtime_error( "Can only sweep multipatch connections of 2d maps." );

            // All constituents have the same orientation in this situation, so use ZeroToZero.
            out.emplace( left_side, std::pair<TPPermutation, ConstituentSide>{ TPPermutation::ZeroToZero, perm_and_right_side.second } );
        }

        return out;
    }

    MultiPatchCombinatorialMap blockLayout( const MultiPatchCombinatorialMap& cmap )
    {
        const std::shared_ptr<const TPCombinatorialMap> single_elem_patch = [&]() {
            const auto cmap1d = std::make_shared<const CombinatorialMap1d>( 1 );
            const auto cmap2d = std::make_shared<const TPCombinatorialMap>( cmap1d, cmap1d );
            if( cmap.dim() == 3 ) {
                return std::make_shared<const TPCombinatorialMap>( cmap2d, cmap1d );
            } else {
                return cmap2d;
            }
        }();

        const MultiPatchCombinatorialMap block_layout(
            std::vector<std::shared_ptr<const TPCombinatorialMap>>( cmap.constituents().size(), single_elem_patch ),
            cmap.connections() );
        return block_layout;
    }
} // namespace topology