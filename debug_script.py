import debugpy

# Check if a client is already connected (optional)
if debugpy.is_client_connected():
    print("A debugger is already connected. Restart VS Code if needed.")

# Start a new debug session
debugpy.listen(("localhost", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
print("Debugger attached.")
