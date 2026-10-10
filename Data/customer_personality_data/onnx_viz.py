"""Shared helper for all `*_pytorch.ipynb`: export a trained oracle to ONNX and
render its forward graph (input -> ... -> output) in a single function call.

PNGs and `.onnx` files go to `media/` and are embedded in REPORT.md.
Open any `media/*.onnx` in Netron to inspect the model interactively.
"""

import torch
import onnx
from onnx import shape_inference
import graphviz


# ONNX has no "Linear" operator: torch exports nn.Linear as Gemm (Y = A@B + C).
# Relabel so the graph reads the way the code does.
OP_LABELS = {"Gemm": "Linear"}


def _tensor_shape(name, model):
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.name == name:
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(str(d.dim_value) if d.HasField("dim_value") else (d.dim_param or "?"))
            return "×".join(dims)
    return ""


def save_graph(model, name, in_dim, outdir="media"):
    """Export `model` to `outdir/<name>.onnx` and render `outdir/<name>.png`.

    `model` must be a trained PyTorch oracle (it is switched to eval mode here).
    `name` e.g. "arch_clf_v1". Returns the `.onnx` path.
    """
    model.eval()
    onnx_path = f"{outdir}/{name}.onnx"
    torch.onnx.export(
        model, torch.randn(1, in_dim), onnx_path,
        input_names=["input"], output_names=["output"],
        dynamo=False, opset_version=17,
    )
    render_graph(name, outdir)
    return onnx_path


def render_graph(name, outdir="media"):
    """Render `outdir/<name>.png` from an already-exported `outdir/<name>.onnx`."""
    onnx_path = f"{outdir}/{name}.onnx"
    om = onnx.load(onnx_path)
    try:
        om = shape_inference.infer_shapes(om)
    except Exception:
        pass
    g = om.graph
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="TB", nodesep="0.25", ranksep="0.35")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="white",
             color="#1B6B72", fontname="monospace", fontsize="9")

    producer = {}
    for i, node in enumerate(g.node):
        for out in node.output:
            producer[out] = f"n{i}"

    for inp in g.input:
        nid = f"in_{inp.name}"
        dot.node(nid, f"input\n{_tensor_shape(inp.name, om)}",
                 fillcolor="#D3E6E4", color="#1B6B72")
        for i, node in enumerate(g.node):
            if inp.name in node.input:
                dot.edge(nid, f"n{i}")
    for i, node in enumerate(g.node):
        outs = [o for o in node.output if o]
        label = OP_LABELS.get(node.op_type, node.op_type)
        dot.node(f"n{i}", f"{label}\n{_tensor_shape(outs[0], om) if outs else ''}")
        for tin in node.input:
            if tin in producer:
                dot.edge(producer[tin], f"n{i}")
    for out in g.output:
        oid = f"out_{out.name}"
        dot.node(oid, f"output\n{_tensor_shape(out.name, om)}",
                 fillcolor="#D3E6E4", color="#1B6B72")
        if out.name in producer:
            dot.edge(producer[out.name], oid)

    dot.render(f"{outdir}/{name}", cleanup=True)
    print(f"rendered {outdir}/{name}.png from {onnx_path}")
