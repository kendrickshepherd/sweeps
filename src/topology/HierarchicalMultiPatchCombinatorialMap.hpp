#pragma once
#include <MultiPatchCombinatorialMap.hpp>
#include <HierarchicalTPCombinatorialMap.hpp>
#include <DartRange.hpp>
#include <map>

namespace topology
{
    /// @brief A hierarchical combinatorial map over multi-patch topologies.
    ///
    class HierarchicalMultiPatchCombinatorialMap : public CombinatorialMap
    {
        public:
        HierarchicalMultiPatchCombinatorialMap( const std::vector<std::shared_ptr<const MultiPatchCombinatorialMap>>& refinement_levels,
                                                const std::vector<std::vector<Cell>>& leaf_elements );
        virtual ~HierarchicalMultiPatchCombinatorialMap() = default;

        virtual std::optional<Dart> phi( const int i, const Dart& d ) const override;

        virtual Dart::IndexType maxDartId() const override;

        virtual uint dim() const override;

        virtual bool iterateDartsWhile( const std::function<bool( const Dart& )>& callback ) const override;

        virtual bool iterateCellsWhile( const uint cell_dim,
                                        const std::function<bool( const Cell& )>& callback ) const override;

        virtual std::optional<IndexingFunc> indexing( const uint cell_dim ) const override;

        virtual std::optional<size_t> cellCount( const uint ) const override;

        std::pair<size_t, Dart> unrefinedAncestorDartOfCell( const Cell& leaf_c ) const;

        const std::vector<std::shared_ptr<const MultiPatchCombinatorialMap>>& refinementLevels() const { return mRefinementLevels; }

        const std::vector<std::shared_ptr<const HierarchicalTPCombinatorialMap>>& constituents() const { return mConstituents; }

        size_t numLevels() const { return mRefinementLevels.size(); }

        const DartRanges& dartRanges() const { return mRanges; }

        Dart toGlobalDart( const size_t level_ii, const Dart& level_dart ) const;

        bool iterateChildren( const Cell& local_cell,
                              const size_t cell_level,
                              const std::function<bool( const Cell& )>& callback ) const;

        bool iterateAncestors( const Dart& global_d,
                               const std::function<bool( const Dart& )>& callback ) const;

        bool isUnrefinedLeafDart( const Dart& d ) const;

        private:
        const std::vector<std::shared_ptr<const MultiPatchCombinatorialMap>> mRefinementLevels;
        std::vector<std::shared_ptr<const HierarchicalTPCombinatorialMap>> mConstituents;
        const DartRanges mRanges;
    };

    std::vector<std::vector<Cell>> leafElements( const HierarchicalMultiPatchCombinatorialMap& cmap );

    /// @brief A leaf element of a hierarchical multipatch cmap, located on its patch and refinement level.
    struct MultiPatchLeafElementInfo
    {
        Cell leaf;
        size_t patch;
        size_t level;
        /// Level-local and patch-local tensor element indices: the 1d dart ids of the patch's level cell.
        SmallVector<size_t, 3> indices;
    };

    /// @brief The patch, level and level-local tensor element indices of a leaf element.
    /// The leaf field preserves the supplied cell handle.
    /// Patch, level and indices are independent of which dart of the leaf is supplied.
    /// @param leaf  A top-dimensional cell given by any of its leaf darts.
    /// @throws std::invalid_argument if leaf is not top-dimensional.
    MultiPatchLeafElementInfo leafElementInfo( const HierarchicalMultiPatchCombinatorialMap& cmap, const Cell& leaf );

    /// @brief leafElementInfo of every leaf element, in iterateCellsWhile order.
    std::vector<MultiPatchLeafElementInfo> leafElementInfos( const HierarchicalMultiPatchCombinatorialMap& cmap );
}; // namespace topology
