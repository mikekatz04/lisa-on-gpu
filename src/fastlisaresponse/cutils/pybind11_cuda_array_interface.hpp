/*
## Copyright (c) 2021, Mikael Twengström
## All rights reserved.
## This file is part of pybind11_cuda_array_interface and is distributed under the
## BSD-3 Clause License. For full terms see the included LICENSE file.
*/

#pragma once

#define PY_SSIZE_T_CLEAN
#include <pybind11/detail/descr.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <iostream>

#include <cuda_runtime.h>

namespace py = pybind11;

namespace caiexcp {
class BaseError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
} // namespace caiexcp

#define DEFINE_ERROR_CLASS(CLASS_NAME)                                                            \
    class CLASS_NAME : public BaseError                                                           \
    {                                                                                             \
    public:                                                                                       \
        using BaseError::BaseError;                                                               \
        static constexpr const char *class_name = #CLASS_NAME;                                    \
    };

namespace caiexcp {
DEFINE_ERROR_CLASS(InterfaceNotImplementedError)
DEFINE_ERROR_CLASS(IncompleteInterfaceError)
DEFINE_ERROR_CLASS(DtypeMismatchError)
DEFINE_ERROR_CLASS(CudaCallError)
DEFINE_ERROR_CLASS(ReadOnlyAccessError)
DEFINE_ERROR_CLASS(EndiannessDetectionError)
DEFINE_ERROR_CLASS(InvalidShapeError)
DEFINE_ERROR_CLASS(InvalidVersionError)
DEFINE_ERROR_CLASS(InvalidCapsuleError)
DEFINE_ERROR_CLASS(ObjectOwnershipError)
DEFINE_ERROR_CLASS(InvalidTypestrError)
DEFINE_ERROR_CLASS(MissingDeleterError)
DEFINE_ERROR_CLASS(SimpleNamespaceError)
DEFINE_ERROR_CLASS(UnRegCudaTypeError)
} // namespace caiexcp

#undef DEFINE_ERROR_CLASS

namespace caiexcp {

template <typename... Args>
void register_errors(py::module &module)
{
    (py::register_exception<Args>(module, Args::class_name), ...);
}

inline void register_custom_cuda_array_interface_exceptions(py::module &module)
{
    register_errors<InterfaceNotImplementedError, IncompleteInterfaceError, DtypeMismatchError,
                    CudaCallError, ReadOnlyAccessError, EndiannessDetectionError,
                    InvalidShapeError, InvalidVersionError, InvalidCapsuleError,
                    ObjectOwnershipError, InvalidTypestrError, MissingDeleterError,
                    SimpleNamespaceError, UnRegCudaTypeError>(module);
}
} // namespace caiexcp

namespace cuerrutil {

inline const char *enum_to_string(const cudaError_t &error) { return cudaGetErrorName(error); }

template <typename T>
inline void throw_on_unsuccessful_enum(T result, const char *func, const char *file, int line)
{
    if (result) {
        std::stringstream ss_error;
        ss_error << "CUDA error at " << file << ":" << line
                 << " code=" << static_cast<unsigned int>(result) << "(" << enum_to_string(result)
                 << ") \"" << func << "\"";

        throw caiexcp::CudaCallError(ss_error.str());
    }
}
} // namespace cuerrutil

#define checkCudaErrors(val) cuerrutil::throw_on_unsuccessful_enum(val, #val, __FILE__, __LINE__)

namespace cai {

// Forward declarations
template <typename T>
class cuda_array_t;

template <typename T>
class cuda_memory_handle
{
private:
    void *ptr;
    std::function<void(void *)> deleter;

    // Constructor for C++-created objects (default deleter)
    cuda_memory_handle(void *ptr)
        : ptr(ptr), deleter([](void *ptr) { checkCudaErrors(cudaFree(ptr)); })
    {
    }

    // Constructor for Python-created objects (explicit do-nothing deleter)
    cuda_memory_handle(void *ptr, std::function<void(void *)> deleter)
        : ptr(ptr), deleter(std::move(deleter))
    {
    }

    friend class cuda_array_t<T>;
    friend class CudaArrayInterfaceTest;
    friend class py::detail::type_caster<cuda_array_t<T>>;

public:
    ~cuda_memory_handle() { deleter(ptr); }

protected:
    // Factory method
    template <typename... Args>
    static std::shared_ptr<cuda_memory_handle> make_shared_handle(Args &&...args)
    {
        return std::shared_ptr<cuda_memory_handle>(
            new cuda_memory_handle(std::forward<Args>(args)...));
    }
};

template <typename T>
class cuda_array_t
{
private:
    std::shared_ptr<cuda_memory_handle<T>> handle;
    std::vector<size_t> shape;
    std::string typestr;
    bool readonly;
    int version;
    py::object py_obj{py::none()};

    void *ptr() const { return handle->ptr; }

    void check_dtype() const
    {
        py::dtype expected_dtype = py::dtype::of<T>();
        py::dtype actual_dype(this->typestr);

        if (!expected_dtype.is(actual_dype)) {
            std::stringstream error_ss;
            error_ss << "Mismatching dtypes. "
                     << "Expected the dtype: " << py::str(expected_dtype).cast<std::string>()
                     << " corresponding"
                     << " to a C++ " << typeid(T).name() << " which is not compatible "
                     << "with the supplied dtype " << py::str(actual_dype).cast<std::string>()
                     << "\n";
            throw caiexcp::DtypeMismatchError(error_ss.str());
        }
    }

    std::string determine_endianness()
    {
        constexpr uint32_t number = 1;
        const auto *bytePtr = reinterpret_cast<const uint8_t *>(&number);

        auto firstByte = *bytePtr;
        auto lastByte = *(bytePtr + sizeof(uint32_t) - 1);

        std::string endiann_str;
        if (firstByte == 1 && lastByte == 0) {
            endiann_str = "<"; // little-endian
        } else if (firstByte == 0 && lastByte == 1) {
            endiann_str = ">"; // big-endian
        } else {
            // endiann_str = "|"; // not-relevant
            caiexcp::EndiannessDetectionError("Unable to determine system endianness");
        }
        return endiann_str;
    }

    cuda_array_t() = default;

    void make_cuda_array_t()
    {
        typestr = determine_endianness() + py::format_descriptor<T>::format()
                  + std::to_string(sizeof(T));
        void *deviceptr;
        checkCudaErrors(cudaMalloc(&deviceptr, size_of_shape() * sizeof(T)));
        handle = cuda_memory_handle<T>::make_shared_handle(deviceptr);
    };

    friend class py::detail::type_caster<cuda_array_t<T>>;

public:
    cuda_array_t(std::vector<size_t> shape, const bool readonly = false, const int version = 3)
        : shape(std::move(shape)), readonly(readonly), version(version)
    {
        this->make_cuda_array_t();
    };

    const std::vector<size_t> &get_shape() const { return shape; }

    py::dtype get_dtype() const { return py::dtype{typestr}; }

    bool is_readonly() const { return readonly; }

    int get_version() const { return version; }

    size_t size_of_shape() const
    {
        return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1),
                               std::multiplies<>());
    }

    T *get_compatible_typed_pointer()
    {
        if (!readonly) {
            check_dtype();
            return reinterpret_cast<T *>(this->ptr());
        }
        throw caiexcp::ReadOnlyAccessError(
            "Attempt to modify instance of cuda_array_t<T> with attribute readonly=true");
    }

    const T *get_compatible_typed_pointer() const
    {
        check_dtype();
        return reinterpret_cast<const T *>(this->ptr());
    }
};

template <typename T>
class cuda_shared_ptr_holder
{
private:
    std::shared_ptr<cuda_memory_handle<T>> holder_ptr;

    cuda_shared_ptr_holder(std::shared_ptr<cuda_memory_handle<T>> sharedPtr)
        : holder_ptr(std::move(sharedPtr))
    {
    }

    // Static factory method that encapsulates a shared_ptr<cuda_memory_handle<T>>
    static cuda_shared_ptr_holder *create(const std::shared_ptr<cuda_memory_handle<T>> &sharedPtr)
    {
        return new cuda_shared_ptr_holder(sharedPtr);
    }

    friend class CudaArrayInterfaceTest;
    friend class py::detail::type_caster<cuda_array_t<T>>;
};

inline void validate_typestr(const std::string &typestr)
{
    if (typestr.length() < 3) {
        throw caiexcp::InvalidTypestrError("Invalid typestr: too short.");
    }

    // Check endianness
    std::set<char> valid_endianness = {'<', '>', '|'};
    if (valid_endianness.find(typestr[0]) == valid_endianness.end()) {
        throw caiexcp::InvalidTypestrError("Invalid typestr: Invalid byte order character.");
    }

    // Check type character code
    std::set<char> valid_type_codes = {'t', 'b', 'i', 'u', 'f', 'c', 'm', 'M', 'O', 'S', 'U', 'V'};
    if (valid_type_codes.find(typestr[1]) == valid_type_codes.end()) {
        throw caiexcp::InvalidTypestrError("Invalid typestr: Invalid type character code.");
    }

    // Check byte size
    std::string byte_size_str = typestr.substr(2);
    try {
        const size_t byte_size = std::stoul(byte_size_str); // convert string to unsigned long
        if (byte_size == 0) {
            throw caiexcp::InvalidTypestrError("Invalid typestr: Byte size cannot be zero.");
        }
    } catch (const std::exception &) {
        throw caiexcp::InvalidTypestrError("Invalid typestr: Invalid byte size.");
    }
}

inline void validate_shape(const std::vector<size_t> &shape_vec)
{
    if (shape_vec.empty()) {
        throw caiexcp::InvalidShapeError("'shape' cannot be empty");
    }
    for (const auto &s_elem : shape_vec) {
        if (s_elem < 1) {
            throw caiexcp::InvalidShapeError("All elements in 'shape' should be non-negative "
                                             "integers in the provided __cuda_array_interface__");
        }
    }
}

inline void validate_cuda_ptr(void *ptr)
{
    cudaPointerAttributes attributes;
    checkCudaErrors(cudaPointerGetAttributes(&attributes, ptr));
    if (attributes.type == cudaMemoryTypeUnregistered) {
        throw caiexcp::UnRegCudaTypeError("Invalid cuda device pointer in cuda_array_t object");
    }
}

template <typename T>
void validate_cuda_memory_handle(const std::shared_ptr<cuda_memory_handle<T>> &sptr)
{
    if (sptr.use_count() != 1) {
        throw caiexcp::ObjectOwnershipError("cuda_memory_handle has invalid reference count!");
    }
}

inline void validate_capsule(const py::capsule &vcaps)
{
    if (!vcaps) {
        throw caiexcp::InvalidCapsuleError("Capsule creation failed.");
    }
    if (std::string(vcaps.name()) != "cuda_memory_capsule") {
        throw caiexcp::InvalidCapsuleError("Capsule has an unexpected name.");
    }
}

} // namespace cai

namespace pybind11::detail {
template <typename T>
class handle_type_name<cai::cuda_array_t<T>>
{
public:
    static constexpr auto name
        = const_name("cai::cuda_array_t[") + npy_format_descriptor<T>::name + const_name("]");
};
} // namespace pybind11::detail

template <typename T>
class py::detail::type_caster<cai::cuda_array_t<T>>
{
public:
    using type = cai::cuda_array_t<T>;
    PYBIND11_TYPE_CASTER(cai::cuda_array_t<T>, py::detail::handle_type_name<type>::name);

    // Python -> C++ conversion
    bool load(py::handle src, bool ___)
    {
        static_cast<void>(___); // Unused is intentional

        auto obj = py::reinterpret_borrow<py::object>(src);

        if (!py::hasattr(obj, "__cuda_array_interface__")) {
            throw caiexcp::InterfaceNotImplementedError(
                "Provided Python Object does not implement __cuda_array_interface__");
        }

        py::object interface = obj.attr("__cuda_array_interface__");
        auto iface_dict = interface.cast<py::dict>();

        std::vector<std::string> mandatory_fields = {"data", "shape", "typestr", "version"};

        for (const auto &field : mandatory_fields) {
            if (!iface_dict.contains(field)) {
                throw caiexcp::IncompleteInterfaceError(
                    "Mandatory field " + field
                    + " is missing from the provided __cuda_array_interface__");
            }
        }

        // Extract the version key from the cuda array dict
        value.version = iface_dict["version"].cast<int>();
        if (value.version != 3) {
            throw caiexcp::InvalidVersionError(
                "Unsupported __cuda_array_interface__ version != 3");
        }

        // Extract the shape and check if all elements are unsigned integers
        auto shape_tuple = iface_dict["shape"].cast<py::tuple>();
        // Extract the shape key from the cuda array dict
        for (py::handle s_elem : shape_tuple) {
            if (py::isinstance<py::int_>(s_elem) && s_elem.cast<ssize_t>() >= 0) {
                value.shape.emplace_back(s_elem.cast<size_t>());
            } else {
                throw caiexcp::InvalidShapeError("'shape' can only contain non-negative integers");
            }
        }
        cai::validate_shape(value.shape);

        // Extract the typestr key from the cuda array dict
        value.typestr = iface_dict["typestr"].cast<std::string>();
        cai::validate_typestr(value.typestr);

        // Extract the data key from the cuda array dict
        auto data = iface_dict["data"].cast<py::tuple>();
        auto ptr_obj = data[0].cast<py::object>();

        void *inputptr{nullptr};
        if (!ptr_obj.is_none()) {
            inputptr = reinterpret_cast<void *>(ptr_obj.cast<uintptr_t>());
        }
        cai::validate_cuda_ptr(inputptr);

        value.handle = cai::cuda_memory_handle<T>::make_shared_handle(inputptr, [](void *) {});
        cai::validate_cuda_memory_handle<T>(value.handle);

        value.readonly = data[1].cast<bool>();

        value.check_dtype();

        // Keep a reference to the original Python object to prevent it from being garbage
        // collected
        value.py_obj = obj;

        return true;
    }

    // C++ -> Python conversion
    static py::handle cast(const cai::cuda_array_t<T> &src, return_value_policy /* policy */,
                           handle /* parent */)
    {

        cai::validate_shape(src.shape);
        cai::validate_typestr(src.typestr);
        src.check_dtype();
        cai::validate_cuda_ptr(src.ptr());
        cai::validate_cuda_memory_handle<T>(src.handle);

        // If py_obj of src is set it means it originates from Python and the object can thus be
        // released.
        if (!src.py_obj.is_none()) {
            return src.py_obj;
        }

        py::dict interface;

        interface["shape"] = py::tuple(py::cast(src.shape));
        interface["typestr"] = py::str(py::cast(src.typestr));
        interface["data"]
            = py::make_tuple(py::int_{reinterpret_cast<uintptr_t>(src.ptr())}, src.readonly);
        interface["version"] = 3;

        // Assuming src was created in C++, src.handle owns the CUDA memory and should
        // be wrapped in a py::capsule to transfer ownership to Python.
        py::capsule caps(cai::cuda_shared_ptr_holder<T>::create(src.handle), "cuda_memory_capsule",
                         [](void *cap_ptr) {
                             delete reinterpret_cast<cai::cuda_shared_ptr_holder<T> *>(cap_ptr);
                         });
        cai::validate_capsule(caps);

        // Create an instance of a Python object that can hold arbitrary attributes
        py::object caio{py::none()};
        try {
            py::object types = py::module::import("types");
            caio = types.attr("SimpleNamespace")();
        } catch (const std::exception &e) {
            throw caiexcp::SimpleNamespaceError(
                "Failed to import the types module and create a SimpleNamespace object.");
        }
        caio.attr("__cuda_array_interface__") = interface;

        // Make the SimpleNamespace object own the capsule to prevent it from being garbage
        // collected.
        caio.attr("_capsule") = caps;

        return caio.release();
    }
};
