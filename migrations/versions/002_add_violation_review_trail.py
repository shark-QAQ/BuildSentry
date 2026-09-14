"""add violation review trail

复核留痕：记录违规项被谁、何时、以什么理由复核。

Revision ID: 002
Revises: 001
Create Date: 2026-09-14

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite 不支持 ADD CONSTRAINT，batch 模式会重建表后再迁移数据
    with op.batch_alter_table("violations") as batch_op:
        batch_op.add_column(sa.Column("reviewed_by", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("review_comment", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.create_foreign_key(
            "fk_violations_reviewed_by_users", "users", ["reviewed_by"], ["id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("violations") as batch_op:
        batch_op.drop_constraint("fk_violations_reviewed_by_users", type_="foreignkey")
        batch_op.drop_column("reviewed_at")
        batch_op.drop_column("review_comment")
        batch_op.drop_column("reviewed_by")
