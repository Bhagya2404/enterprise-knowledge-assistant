from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader

loader = DirectoryLoader(
    "Data",
    glob="*.docx",
    loader_cls=Docx2txtLoader
)

documents = loader.load()

print("Number of documents:", len(documents))

for doc in documents:
    print("\nFile:", doc.metadata["source"])
    print("\nPreview:")
    print(doc.page_content[:200])