vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO realtimechris/benchmarksuite
    REF "v${VERSION}"    
    SHA512 482849d0825d0ba3136a2ab056e5eed9326d0cf869832d37e7bfeb8e34c98bcd5641199a1773f334657d62ade431740354574e818ef4d19a5ecc099f9fcecec5
    HEAD_REF main
)

set(VCPKG_BUILD_TYPE release) # header-only

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/License.md")
