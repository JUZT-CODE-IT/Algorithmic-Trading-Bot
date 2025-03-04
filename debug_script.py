import debugpy

# Disconnect previous debug sessions
if debugpy.is_client_connected():
    debugpy.disconnect()
    print("Previous debugger session closed.")

# Start a new debug session
debugpy.listen(("localhost", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
print("Debugger attached.")
