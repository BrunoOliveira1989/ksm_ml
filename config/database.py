from sqlalchemy import create_engine

DATABASE_URL = (
    "postgresql+psycopg2://kodiak_pocket_owner:"
    "k0rm1fPEwAyU@ep-long-surf-a5z0iq90-pooler."
    "us-east-2.aws.neon.tech/ksm?sslmode=require"
)
engine = create_engine(DATABASE_URL)
