from google.auth import default

credentials, project = default()

print("Credentials type:", type(credentials))
print("Project:", project)
