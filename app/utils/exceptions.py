"""
exceptions.py — All typed custom exceptions for Truth Engine.
Created FIRST per architecture blueprint.
Extends the original RAG-Lite exceptions with multi-source and format-specific ones.
"""

# ── File Parsing Exceptions ───────────────────────────────────────────────────

class FileParseError(Exception):
    """Generic unrecoverable parse failure for any file type."""
    def __init__(self, message: str = "Failed to parse the file."):
        super().__init__(message)
        self.message = message

class PDFEncryptedError(FileParseError):
    def __init__(self, message: str = "This PDF is password-protected. Please remove the password and re-upload."):
        super().__init__(message)
        self.message = message

class PDFNoTextError(FileParseError):
    def __init__(self, message: str = "No extractable text found. This appears to be a scanned PDF."):
        super().__init__(message)
        self.message = message

class PDFParseError(FileParseError):
    def __init__(self, message: str = "Failed to parse the PDF file. It may be corrupted or unsupported."):
        super().__init__(message)
        self.message = message

class JSONParseError(FileParseError):
    def __init__(self, message: str = "Failed to parse the JSON/CSV file. Check the format."):
        super().__init__(message)
        self.message = message

class MarkdownParseError(FileParseError):
    def __init__(self, message: str = "Failed to parse the Markdown file."):
        super().__init__(message)
        self.message = message

class UnsupportedFileTypeError(FileParseError):
    def __init__(self, ext: str = ""):
        msg = f"Unsupported file type: '{ext}'. Accepted: PDF, JSON, CSV, MD, TXT."
        super().__init__(msg)
        self.message = msg

# ── Source Exceptions ─────────────────────────────────────────────────────────

class SourceNotLoadedError(Exception):
    """Raised when a query is attempted against a source that has no index yet."""
    def __init__(self, source: str):
        msg = f"Source {source} has not been loaded. Please upload a file for it first."
        super().__init__(msg)
        self.message = msg

# ── Chunking Exceptions ───────────────────────────────────────────────────────

class ChunkingError(Exception):
    def __init__(self, message: str = "Could not split document into chunks."):
        super().__init__(message)
        self.message = message

# ── Embedding Exceptions ──────────────────────────────────────────────────────

class EmbeddingError(Exception):
    def __init__(self, message: str = "Embedding model failed to process the input."):
        super().__init__(message)
        self.message = message

class EmptyEmbeddingInputError(EmbeddingError):
    def __init__(self, message: str = "Cannot embed empty input."):
        super().__init__(message)
        self.message = message

# ── Vector Store Exceptions ───────────────────────────────────────────────────

class IndexNotReadyError(Exception):
    def __init__(self, message: str = "Vector index is not ready. Please upload a document first."):
        super().__init__(message)
        self.message = message

class IndexBuildError(Exception):
    def __init__(self, message: str = "Failed to build the vector index."):
        super().__init__(message)
        self.message = message

# ── LLM Exceptions ────────────────────────────────────────────────────────────

class LLMProviderError(Exception):
    def __init__(self, provider: str, message: str):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.message = message

class LLMRateLimitError(LLMProviderError):
    pass

class LLMProviderUnavailableError(LLMProviderError):
    pass

class LLMTimeoutError(LLMProviderError):
    pass

class FatalLLMError(Exception):
    def __init__(self, message: str = "All LLM providers are currently unavailable."):
        super().__init__(message)
        self.message = message

# ── Session Exceptions ────────────────────────────────────────────────────────

class SessionNotFoundError(Exception):
    def __init__(self, session_id: str):
        msg = f"Session '{session_id}' not found. Please re-upload your documents."
        super().__init__(msg)
        self.session_id = session_id
        self.message = msg
