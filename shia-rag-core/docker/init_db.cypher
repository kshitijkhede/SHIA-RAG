// ============================================================
// SHIA-RAG Neo4j Schema Initialization
// Run via: cypher-shell -u neo4j -p <password> < init_db.cypher
// ============================================================

// ── Constraints: Enforce unique node IDs ──
CREATE CONSTRAINT kn_node_id IF NOT EXISTS
FOR (n:KnowledgeNode) REQUIRE n.node_id IS UNIQUE;

CREATE CONSTRAINT doc_block_id IF NOT EXISTS
FOR (b:DocumentBlock) REQUIRE b.block_id IS UNIQUE;

CREATE CONSTRAINT domain_root_id IF NOT EXISTS
FOR (r:DomainRoot) REQUIRE r.root_id IS UNIQUE;

// ── Indexes: Accelerate lookups ──
CREATE INDEX kn_canonical_name IF NOT EXISTS
FOR (n:KnowledgeNode) ON (n.canonical_name);

CREATE INDEX kn_domain IF NOT EXISTS
FOR (n:KnowledgeNode) ON (n.domain);

CREATE INDEX doc_block_doc_id IF NOT EXISTS
FOR (b:DocumentBlock) ON (b.doc_id);

// ── Example Seed Data (Optional, remove for production) ──
// CREATE (:DomainRoot {root_id: "ROOT-NET", domain: "Networking", created_at: datetime()})
// CREATE (:DomainRoot {root_id: "ROOT-OS", domain: "Operating Systems", created_at: datetime()})

RETURN "SHIA-RAG Neo4j schema initialized successfully" AS status;
