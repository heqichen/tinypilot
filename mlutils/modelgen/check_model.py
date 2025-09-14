import onnx

# Define the path to your ONNX model
model_path = "outptu.onnx"

try:
    # Load the ONNX model
    onnx_model = onnx.load(model_path)

    # Check the model for validity
    onnx.checker.check_model(onnx_model)

    print("ONNX model is valid.")

except onnx.checker.ValidationError as e:
    print(f"ONNX model is invalid: {e}")

except Exception as e:
    print(f"An error occurred while checking the ONNX model: {e}")
