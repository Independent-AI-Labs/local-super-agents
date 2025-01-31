# TODO Will be replaced by Apache Tika.
TEXT_BASED_EXTENSIONS = [
    # Plain text and basic formats
    '.txt', '.text', '.log', '.md', '.markdown',

    # Structured data formats
    '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.csv', '.tsv', '.psv',  # Comma, Tab, and Pipe Separated Values

    # Source code
    '.py', '.pyx', '.pyi',  # Python
    '.js', '.jsx', '.ts', '.tsx',  # JavaScript and TypeScript
    '.html', '.htm', '.css', '.scss', '.sass', '.less',  # Web
    '.c', '.cpp', '.cxx', '.h', '.hpp',  # C and C++
    '.java', '.kt', '.ktm', '.kts',  # Java and Kotlin
    '.cs', '.fs', '.vb',  # .NET languages
    '.go',  # Go
    '.rs',  # Rust
    '.rb', '.erb',  # Ruby
    '.php',  # PHP
    '.swift',  # Swift
    '.scala',  # Scala
    '.groovy',  # Groovy
    '.pl', '.pm',  # Perl
    '.sh', '.bash', '.zsh',  # Shell env
    '.lua',  # Lua
    '.r', '.R',  # R
    '.m',  # MATLAB/Octave
    '.sql',  # SQL

    # Markup and documentation
    '.rst', '.tex', '.latex',
    '.adoc', '.asciidoc',  # AsciiDoc

    # Config and build files
    '.properties', '.env',
    '.gitignore', '.dockerignore',
    '.gradle',
    '.cmake', 'CMakeLists.txt',
    'Makefile', '.mk',
    'Dockerfile',
    '.htaccess',

    # Other structured formats
    '.svg',  # SVG is text-based XML
    '.graphql', '.gql',  # GraphQL
    '.proto',  # Protocol Buffers
    '.plist',  # Property list (often used in Apple ecosystems)
    '.wsdl', '.xsd',  # Web Services Description Language and XML Schema

    # Package management
    'package.json', 'Gemfile', 'requirements.txt', 'Cargo.toml', 'pom.xml',

    # Specific config files (without extension)
    'Jenkinsfile', 'Vagrantfile', '.babelrc', '.eslintrc'
]
