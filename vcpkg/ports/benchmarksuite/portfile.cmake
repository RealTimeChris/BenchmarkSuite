vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO realtimechris/benchmarksuite
    REF "v${VERSION}"    
    SHA512 f61f8f673a7ba40ac94b3b86e7c007d5cecf351bec3cbe4d0c4458ff86da840ec244b65fc47c6bed75c1b6a431d270d5796ed1fecc50b0d3cbcdab21d0216d29
    HEAD_REF main
)

set(VCPKG_BUILD_TYPE release) # header-only

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/License.md")
