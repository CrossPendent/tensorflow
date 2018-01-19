package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

py_library(
    name = "grip_input_data",
    srcs = [
        "grip_input_data.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "grip_input_data_test",
    size = "small",
    srcs = ["grip_input_data_test.py"],
    additional_deps = [
        ":grip_input_data",
    ],
)

py_binary(
    name = "tf_practice",
    srcs = [
        "tf_practice.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":input_data",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
