"""
中文：模型检查工具。递归提取 PyTorch 模型的层级结构信息，用于前端架构可视化。
English: Model inspection utilities. Recursively extracts PyTorch model hierarchy information for frontend architecture visualization.
"""


def extract_module_info(module, module_name=""):
    """
    中文：递归提取 PyTorch nn.Module 的层级结构信息。
    English: Recursively extract information from a PyTorch nn.Module hierarchy.

    Args:
        module: PyTorch nn.Module 对象 / PyTorch nn.Module object
        module_name: 模块名称（用于显示）/ Module name (for display)

    Returns:
        包含模块结构信息的字典 / Dictionary containing module structure information
    """
    # Get class name
    class_name = module.__class__.__name__

    # Build result dictionary
    result = {
        "name": module_name,
        "class": class_name,
        "details": "",
        "children": []
    }

    # Extract relevant details based on module type
    if class_name == "ParallelCNNBlock":
        details = f"Kernels: {module.kernel_sizes}, Hidden: {module.hidden_dim}"
        result["details"] = details

        # Add convolution branches as children
        for i, conv in enumerate(module.conv_branches):
            conv_info = {
                "name": f"conv_{i}",
                "class": "Conv1d",
                "details": f"kernel_size={module.kernel_sizes[i]}, out_channels={module.hidden_dim}",
                "children": []
            }
            result["children"].append(conv_info)

    elif class_name == "GCNBlock":
        details = f"Layers: {module.num_layers}, Hidden: {module.hidden_dim}, Out: {module.out_channels}"
        result["details"] = details

        # Add GCN layers as children
        for i, gcn_layer in enumerate(module.gcn_layers):
            gcn_info = {
                "name": f"gcn_layer_{i}",
                "class": "GCNConv",
                "details": f"layer_{i}",
                "children": []
            }
            result["children"].append(gcn_info)

    elif class_name == "ClassQueryHead":
        details = f"Classes: {module.num_classes}, Heads: {module.num_heads if hasattr(module, 'num_heads') else 'N/A'}"
        result["details"] = details

        # Add class queries as children
        for i in range(module.num_classes):
            query_info = {
                "name": f"class_query_{i}",
                "class": "LearnableQuery",
                "details": f"class_{i}",
                "children": []
            }
            result["children"].append(query_info)

    elif class_name == "ClassQueryHeadPooling":
        details = f"Classes: {module.num_classes}"
        result["details"] = details

        # Add class queries as children
        for i in range(module.num_classes):
            query_info = {
                "name": f"class_query_{i}",
                "class": "LearnableQuery",
                "details": f"class_{i}",
                "children": []
            }
            result["children"].append(query_info)

    elif class_name == "HierarchicalClassQueryHeadPooling":
        details = f"Classes: {module.num_classes}, Groups: {module.num_groups}"
        result["details"] = details

        # Add group queries as children
        for i, group_name in enumerate(module.group_names):
            group_info = {
                "name": f"group_query_{i}",
                "class": "GroupQuery",
                "details": f"{group_name} group",
                "children": []
            }

            # Add derived class queries
            if group_name in module.group_to_class_indices:
                class_indices = module.group_to_class_indices[group_name]
                for j, class_idx in enumerate(class_indices):
                    class_info = {
                        "name": f"class_query_{class_idx}",
                        "class": "DerivedClassQuery",
                        "details": f"class_{class_idx}",
                        "children": []
                    }
                    group_info["children"].append(class_info)

            result["children"].append(group_info)

    # Recursively process named children
    for name, child in module.named_children():
        if not child.__class__.__name__ in ["Sequential", "ModuleList", "ModuleDict"]:
            child_info = extract_module_info(child, name)
            result["children"].append(child_info)

    return result
