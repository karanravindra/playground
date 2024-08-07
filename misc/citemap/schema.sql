-- DROP TABLE IF EXISTS papers;
-- DROP TABLE IF EXISTS paper_references;
CREATE TABLE
    IF NOT EXISTS papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        paperId VARCHAR(255) UNIQUE,
        corpusId INT UNIQUE,
        title TEXT,
        abstract TEXT,
        url TEXT,
        year INT,
        referenceCount INT,
        citationCount INT,
        influentialCitationCount INT,
        read BOOLEAN
    );

CREATE TABLE
    IF NOT EXISTS paper_references (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id INT REFERENCES papers (id) ON DELETE CASCADE,
        referenced_paper_id INT REFERENCES papers (id) ON DELETE CASCADE
    );