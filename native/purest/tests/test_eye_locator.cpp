#include "eye_locator.hpp"
#include <stdexcept>

void require(bool condition) {
    if (!condition) throw std::runtime_error("eye-pair geometry regression");
}

int main() {
    using plr::EyeLocator;
    require(EyeLocator::plausible_pair({40, 30, 40, 40}, {130, 32, 42, 42}));
    require(EyeLocator::plausible_pair({130, 32, 42, 42}, {40, 30, 40, 40}));
    require(!EyeLocator::plausible_pair({40, 30, 40, 40}, {45, 32, 40, 40}));
    require(!EyeLocator::plausible_pair({40, 30, 40, 40}, {130, 90, 42, 42}));
    require(!EyeLocator::plausible_pair({40, 30, 40, 40}, {130, 32, 15, 15}));
    require(!EyeLocator::plausible_pair({10, 30, 20, 20}, {150, 32, 20, 20}));
    return 0;
}
