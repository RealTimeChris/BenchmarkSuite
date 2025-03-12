#pragma once

#include <oiml/common/oiml_ring_buffer.hpp>
#include <oiml/common/array.hpp>
#include <filesystem>
#include <fstream>

namespace oiml {

	template<size_t alignment> class oiml_file_loader {
	  public:
		static oiml_ring_buffer<char, alignment> loadFile(const std::string& filePath) {
			if (!std::filesystem::exists(filePath)) {
				std::ofstream createFile{ filePath, std::ios::binary };
				createFile.close();
			}

			std::ifstream theStream{ filePath, std::ios::binary | std::ios::ate };
			if (!theStream) {
				throw std::runtime_error("Failed to open file.");
			}

			std::streamsize size = theStream.tellg();
			theStream.seekg(0, std::ios::beg);

			oiml_ring_buffer<char, alignment> buffer(size);
			if (!theStream.read(buffer.claim_space(size), size)) {
				throw std::runtime_error("Failed to read file.");
			}

			theStream.close();
			return buffer;
		}

		static void saveFile(const std::string& fileToSave, const std::string& filePath, bool retry = true) {
			std::ofstream theStream{ filePath.data(), std::ios::binary | std::ios::out | std::ios::trunc };
			theStream.write(fileToSave.data(), static_cast<int64_t>(fileToSave.size()));
			if (theStream.is_open()) {
				std::cout << "File succesfully written to: " << filePath << std::endl;
			} else {
				std::string directory{ filePath.substr(0, filePath.find_last_of("/") + 1) };
				if (!std::filesystem::exists(directory) && retry) {
					std::filesystem::create_directories(directory);
					return saveFile(fileToSave, filePath, false);
				}
				std::cerr << "File failed to be written to: " << filePath << std::endl;
			}
			theStream.close();
		}
	};

}