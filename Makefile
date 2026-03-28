BUILD_DIR := build
CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=Release
NPROC := $(shell nproc)

.PHONY: prepare build test clean rebuild

## 配置 cmake（首次或 CMakeLists.txt 变更后执行）
prepare:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. $(CMAKE_FLAGS)

## 编译
build: | prepare
	@cmake --build $(BUILD_DIR) -j$(NPROC)

## 运行测试
test: build
	@cd $(BUILD_DIR) && ctest --output-on-failure

## 清理构建产物
clean:
	@rm -rf $(BUILD_DIR)

## 完全重建
rebuild: clean build
