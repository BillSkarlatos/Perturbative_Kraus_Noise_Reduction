from qiskit_ibm_runtime import QiskitRuntimeService

# Initialize the service
service = QiskitRuntimeService()

# List available backends
available_backends = service.backends()

# Print backend names
print("Available backends:", [backend.name for backend in available_backends])
