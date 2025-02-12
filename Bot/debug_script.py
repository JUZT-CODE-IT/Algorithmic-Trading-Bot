import debugpy

# Start debugpy server
debugpy.listen(('localhost', 5678))
print("Waiting for debugger to attach...")

# Pause here until debugger attaches
debugpy.wait_for_client()

# Your regular code below
print("Debugger attached!")
