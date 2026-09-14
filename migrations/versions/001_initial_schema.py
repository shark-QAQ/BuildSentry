"""initial schema

Revision ID: 001
Revises: 
Create Date: 2026-09-08

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
# SQLite 只把 "INTEGER PRIMARY KEY" 当作 rowid 别名并自动赋值，BIGINT 主键不会自增。
# 此处与 app/models/db_models.py 的 BigIntPK 保持一致。
BigIntPK = sa.BigInteger().with_variant(sa.Integer(), "sqlite")

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users 表
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('username', sa.String(64), nullable=False),
        sa.Column('email', sa.String(128), nullable=False),
        sa.Column('hashed_password', sa.String(256), nullable=False),
        sa.Column('full_name', sa.String(128), nullable=True),
        sa.Column('role', sa.String(32), server_default='viewer', nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='1', nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )
    op.create_index('ix_users_username', 'users', ['username'])
    op.create_index('ix_users_email', 'users', ['email'])

    # Documents 表
    op.create_table(
        'documents',
        sa.Column('id', BigIntPK, autoincrement=True, nullable=False),
        sa.Column('file_name', sa.String(255), nullable=False),
        sa.Column('file_hash', sa.String(64), nullable=True),
        sa.Column('file_size', sa.BigInteger(), nullable=True),
        sa.Column('raw_text', sa.Text(), nullable=True),
        sa.Column('word_count', sa.Integer(), nullable=True),
        sa.Column('title', sa.String(255), nullable=True),
        sa.Column('project_name', sa.String(255), nullable=True),
        sa.Column('company', sa.String(255), nullable=True),
        sa.Column('status', sa.String(32), server_default='parsed', nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Audit Tasks 表
    op.create_table(
        'audit_tasks',
        sa.Column('id', BigIntPK, autoincrement=True, nullable=False),
        sa.Column('task_no', sa.String(32), nullable=False),
        sa.Column('document_id', BigIntPK, nullable=False),
        sa.Column('status', sa.String(32), server_default='queued', nullable=True),
        sa.Column('progress', sa.Integer(), server_default='0', nullable=True),
        sa.Column('current_step', sa.String(128), nullable=True),
        sa.Column('result_id', BigIntPK, nullable=True),
        sa.Column('queued_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('task_no')
    )

    # Audit Results 表
    op.create_table(
        'audit_results',
        sa.Column('id', BigIntPK, autoincrement=True, nullable=False),
        sa.Column('task_id', BigIntPK, nullable=False),
        sa.Column('conclusion', sa.String(32), nullable=False),
        sa.Column('conclusion_detail', sa.Text(), nullable=True),
        sa.Column('high_risk_count', sa.Integer(), server_default='0', nullable=True),
        sa.Column('medium_risk_count', sa.Integer(), server_default='0', nullable=True),
        sa.Column('low_risk_count', sa.Integer(), server_default='0', nullable=True),
        sa.Column('total_violations', sa.Integer(), server_default='0', nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('model_version', sa.String(64), nullable=True),
        sa.Column('audit_duration_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Violations 表
    op.create_table(
        'violations',
        sa.Column('id', BigIntPK, autoincrement=True, nullable=False),
        sa.Column('task_id', BigIntPK, nullable=False),
        sa.Column('result_id', BigIntPK, nullable=True),
        sa.Column('clause_code', sa.String(64), nullable=True),
        sa.Column('clause_content', sa.Text(), nullable=True),
        sa.Column('location_ref', sa.String(255), nullable=True),
        sa.Column('violation_text', sa.Text(), nullable=True),
        sa.Column('risk_level', sa.String(16), nullable=False),
        sa.Column('violation_type', sa.String(64), nullable=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('suggestion', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('human_review_status', sa.String(32), server_default='pending', nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('violations')
    op.drop_table('audit_results')
    op.drop_table('audit_tasks')
    op.drop_table('documents')
    op.drop_index('ix_users_email', table_name='users')
    op.drop_index('ix_users_username', table_name='users')
    op.drop_table('users')
