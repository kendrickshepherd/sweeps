#pragma once
#include <HierarchicalTPSplineSpace.hpp>
#include <HierarchicalMultiPatchSplineSpace.hpp>
#include <IndexOperations.hpp>
#include <optional>
#include <stdexcept>

namespace basis
{
    /// @brief A level cell refined at the next level that no level function supported in the refined cells covers.
    struct NFormSupportFailure
    {
        size_t patch;
        size_t level;
        /// Level-local tensor cell indices, one per parametric direction.
        util::IndexVec cell;

        bool operator==( const NFormSupportFailure& ) const = default;
    };

    /// @brief A valid space the n-form support check does not implement.
    class NFormSupportNotImplemented : public std::logic_error
    {
        public:
        using std::logic_error::logic_error;
    };

    /// @brief The n-form support policy P_n of ss, the n-form space of a hierarchical complex.
    /// On each level l below the finest, R_l is the set of level l cells with children at level l + 1 and U_l the
    /// union of the supports of the level l functions, active or not, supported in R_l.  P_n holds iff U_l = R_l
    /// on every such level.
    /// @return Empty iff P_n holds; otherwise the first cell of some R_l \ U_l, by level, then by util::flatten
    ///         over the level's cell counts.  patch is 0.
    /// @throws NFormSupportNotImplemented if ss is not 2d.
    std::optional<NFormSupportFailure> checkNFormSupportPolicy( const HierarchicalTPSplineSpace& ss );

    /// @brief P_n on every patch.
    /// @return Empty iff P_n holds on every patch; otherwise the first failure by patch, then level, then cell.
    /// @throws NFormSupportNotImplemented if ss is not 2d.
    std::optional<NFormSupportFailure> checkNFormSupportPolicy( const HierarchicalMultiPatchSplineSpace& ss );
}
