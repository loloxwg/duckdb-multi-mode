#define DUCKDB_EXTENSION_MAIN

#include "onnx_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

namespace duckdb {

inline void OnnxScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vector = args.data[0];
	auto &list_vector = args.data[1];

	auto &source_child = ListVector::GetEntry(list_vector);
	auto &result_child = ListVector::GetEntry(result);

	idx_t current_size = ListVector::GetListSize(result);
	BinaryExecutor::Execute<string_t, list_entry_t, list_entry_t>(
		str_vector,
		list_vector,
		result,
		args.size(),
		[&](string_t path, list_entry_t list_input) {
			idx_t new_size = current_size;
			idx_t result_length = list_input.length;
			ListVector::Reserve(result, new_size);
			list_entry_t result_list;
			result_list.offset = current_size;
			result_list.length = result_length;
			VectorOperations::Copy(source_child, result_child, list_input.offset + list_input.length,
			                       list_input.offset, current_size);
			current_size += list_input.length;
			return result_list;
		});

	ListVector::SetListSize(result, current_size);
}

unique_ptr<FunctionData> OnnxBindFunction(ClientContext &, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	switch (arguments[1]->return_type.id()) {
	case LogicalTypeId::UNKNOWN:
		throw ParameterNotResolvedException();
	case LogicalTypeId::LIST:
		break;
	default:
		throw NotImplementedException("onnx(string, list) requires a list as parameter");
	}
	bound_function.arguments[1] = arguments[1]->return_type;
	bound_function.return_type = arguments[1]->return_type;
	return nullptr;
}

static void LoadInternal(DatabaseInstance &instance) {

	// Register a scalar function
	auto onnx_scalar_function = ScalarFunction("onnx", {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::ANY)},
	                                           LogicalType::LIST(LogicalType::ANY), OnnxScalarFun, OnnxBindFunction);
	ExtensionUtil::RegisterFunction(instance, onnx_scalar_function);
}

void OnnxExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}

std::string OnnxExtension::Name() {
	return "onnx";
}

std::string OnnxExtension::Version() const {
#ifdef EXT_VERSION_ONNX
	return EXT_VERSION_ONNX;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void onnx_init(duckdb::DatabaseInstance &db) {
	duckdb::DuckDB db_wrapper(db);
	db_wrapper.LoadExtension<duckdb::OnnxExtension>();
}

DUCKDB_EXTENSION_API const char *onnx_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif