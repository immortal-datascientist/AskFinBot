# # gdrive_uploader.py using PyDrive
# import os
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive

# # === Step 1: Authenticate ===
# gauth = GoogleAuth()

# # Use saved credentials if available
# cred_file = "mycreds.txt"
# if os.path.exists(cred_file):
#     gauth.LoadCredentialsFile(cred_file)

# if gauth.credentials is None:
#     gauth.LoadClientConfigFile("credentials.json")
#     gauth.LocalWebserverAuth()
# elif gauth.access_token_expired:
#     gauth.Refresh()
# else:
#     gauth.Authorize()

# gauth.SaveCredentialsFile(cred_file)
# drive = GoogleDrive(gauth)

# # === Step 2: Upload Folder/File Structure ===
# def create_or_get_folder(folder_name, parent_id=None):
#     query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
#     if parent_id:
#         query += f" and '{parent_id}' in parents"

#     file_list = drive.ListFile({'q': query}).GetList()
#     if file_list:
#         return file_list[0]['id']

#     folder_metadata = {
#         'title': folder_name,
#         'mimeType': 'application/vnd.google-apps.folder'
#     }
#     if parent_id:
#         folder_metadata['parents'] = [{'id': parent_id}]

#     folder = drive.CreateFile(folder_metadata)
#     folder.Upload()
#     return folder['id']

# def upload_file(file_path, parent_id):
#     file_name = os.path.basename(file_path)
#     file_drive = drive.CreateFile({
#         'title': file_name,
#         'parents': [{'id': parent_id}]
#     })
#     file_drive.SetContentFile(file_path)
#     file_drive.Upload()
#     return file_drive['id']

# def upload_folder(local_folder, parent_id):
#     uploaded_files = []
#     for root, _, files in os.walk(local_folder):
#         rel_path = os.path.relpath(root, local_folder)
#         path_parts = rel_path.split(os.sep) if rel_path != '.' else []

#         current_parent = parent_id
#         for folder in path_parts:
#             current_parent = create_or_get_folder(folder, parent_id=current_parent)

#         for file_name in files:
#             file_path = os.path.join(root, file_name)
#             file_id = upload_file(file_path, current_parent)
#             uploaded_files.append((file_name, file_id))
#     return uploaded_files

# def upload_pdf_data_to_gdrive(base_name):
#     local_dir = os.path.join("stored_pdfs", base_name)
#     root_folder_id = create_or_get_folder("FINACLE_PDF_STORAGE")
#     pdf_folder_id = create_or_get_folder(base_name, parent_id=root_folder_id)
#     return upload_folder(local_dir, parent_id=pdf_folder_id)

# # === Example call ===
# if __name__ == "__main__":
#     uploaded = upload_pdf_data_to_gdrive("example_pdf_name")  # change to your PDF name
#     for name, fid in uploaded:
#         print(f"Uploaded: {name} (ID: {fid})")


import os
import pickle
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# === Authentication ===
gauth = GoogleAuth()
cred_file = "mycreds.txt"

# Clear previous credentials if they caused trouble
if os.path.exists(cred_file):   ###remove
    os.remove(cred_file)      ##remove

# if gauth.credentials is None:
#     gauth.LoadClientConfigFile("credentials.json")  
#     gauth.LocalWebserverAuth()
# elif gauth.access_token_expired:
#     gauth.Refresh()
# else:
#     gauth.Authorize()

# Load config
gauth.LoadClientConfigFile("credentials.json")  # OAuth client (desktop app)

# Start browser flow or manual
try:
    gauth.LocalWebserverAuth()
except Exception as e:
    print("LocalWebserverAuth failed, falling back to CommandLineAuth")
    gauth.CommandLineAuth()

gauth.SaveCredentialsFile(cred_file)
drive = GoogleDrive(gauth)

# === Folder management ===
def create_or_get_folder(folder_name, parent_id=None):
    query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    file_list = drive.ListFile({'q': query}).GetList()
    if file_list:
        return file_list[0]['id']

    folder_metadata = {
        'title': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        folder_metadata['parents'] = [{'id': parent_id}]
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    return folder['id']

# === Uploading ===
def upload_file(file_path, parent_id):
    file_name = os.path.basename(file_path)
    file_drive = drive.CreateFile({'title': file_name, 'parents': [{'id': parent_id}]})
    file_drive.SetContentFile(file_path)
    file_drive.Upload()
    return file_drive['id']

def upload_folder(local_folder, parent_id):
    uploaded_files = []
    for root, _, files in os.walk(local_folder):
        rel_path = os.path.relpath(root, local_folder)
        path_parts = rel_path.split(os.sep) if rel_path != '.' else []

        current_parent = parent_id
        for folder in path_parts:
            current_parent = create_or_get_folder(folder, parent_id=current_parent)

        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_id = upload_file(file_path, current_parent)
            uploaded_files.append((file_name, file_id))
    return uploaded_files

def upload_pdf_data_to_gdrive(base_name, pdf_path, vector_folder, image_map, table_map, image_dir, table_dir):
    root_id = create_or_get_folder("FINACLE_PDF_STORAGE")
    pdf_folder_id = create_or_get_folder(base_name, parent_id=root_id)

    # Upload original PDF
    upload_file(pdf_path, pdf_folder_id)

    # Upload vector .pkl files
    for file in os.listdir(vector_folder):
        if file.endswith(".pkl"):
            upload_file(os.path.join(vector_folder, file), pdf_folder_id)

    # Save and upload image map
    image_map_path = os.path.join(image_dir, "image_map.pkl")
    with open(image_map_path, "wb") as f:
        pickle.dump(image_map, f)
    upload_file(image_map_path, pdf_folder_id)

    # Save and upload table map
    table_map_path = os.path.join(table_dir, "table_map.pkl")
    with open(table_map_path, "wb") as f:
        pickle.dump(table_map, f)
    upload_file(table_map_path, pdf_folder_id)

    # Upload extracted images and tables
    upload_folder(image_dir, pdf_folder_id)
    upload_folder(table_dir, pdf_folder_id)

    return [f['title'] for f in drive.ListFile({'q': f"'{pdf_folder_id}' in parents and trashed=false"}).GetList()]

# === Loading ===
def get_pdf_folder_id(pdf_name):
    root_id = create_or_get_folder("FINACLE_PDF_STORAGE")
    return create_or_get_folder(pdf_name, parent_id=root_id)

def download_file_by_name(name, parent_id):
    query = f"title='{name}' and '{parent_id}' in parents and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()
    if file_list:
        file_drive = file_list[0]
        file_drive.GetContentFile(name)
        return name
    return None

def load_vectors_from_drive(pdf_name):
    folder_id = get_pdf_folder_id(pdf_name)
    for file in drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList():
        if file['title'] == "faiss_index.pkl":
            file.GetContentFile("temp_faiss.pkl")
            with open("temp_faiss.pkl", "rb") as f:
                return pickle.load(f)
    return None

def load_image_map_from_drive(pdf_name):
    folder_id = get_pdf_folder_id(pdf_name)
    downloaded = download_file_by_name("image_map.pkl", folder_id)
    if downloaded:
        with open(downloaded, "rb") as f:
            return pickle.load(f)
    return {}

def load_table_map_from_drive(pdf_name):
    folder_id = get_pdf_folder_id(pdf_name)
    downloaded = download_file_by_name("table_map.pkl", folder_id)
    if downloaded:
        with open(downloaded, "rb") as f:
            return pickle.load(f)
    return {}

def list_drive_pdfs():
    root_id = create_or_get_folder("FINACLE_PDF_STORAGE")
    folders = drive.ListFile({
        'q': f"'{root_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()
    return [f['title'] for f in folders]


