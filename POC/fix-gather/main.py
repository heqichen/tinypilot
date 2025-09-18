from onnx import helper, numpy_helper, TensorProto

import onnx
import numpy as np
import sys
import os


def order_repeated_field(repeated_proto, key_name, order):
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))


def make_node(
    op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs
):
    node = helper.make_node(
        op_type, inputs, outputs, name, doc_string, domain, **kwargs
    )
    if doc_string == "":
        node.doc_string = ""
    order_repeated_field(node.attribute, "name", kwargs.keys())
    return node


def make_graph(*args, doc_string=None, **kwargs):
    graph = helper.make_graph(*args, doc_string=doc_string, **kwargs)
    if doc_string == "":
        graph.doc_string = ""
    return graph


model = helper.make_model(
    opset_imports=[helper.make_operatorsetid("", 14)],
    ir_version=7,
    producer_name="heqichen",
    producer_version="0.0.1",
    graph=make_graph(
        name="main_graph",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, shape=[3, 4, 5])
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, shape=[3, 5])
        ],
        initializer=[],
        nodes=[
            make_node(
                "Constant",
                inputs=[],
                outputs=["onnx::Gather_001"],
                name="Constant_18",
                value=numpy_helper.from_array(np.array([1], dtype="int64"), name=""),
            ),
            make_node(
                "Gather",
                inputs=["input", "onnx::Gather_001"],
                outputs=["output"],
                name="/Gather",
                axis=1,
            ),
        ],
    ),
)

if __name__ == "__main__":
    onnx.save(model, "output.onnx")
