from sqlalchemy.orm import Session
from . import models, schemas

def get_rule(db: Session, rule_id: int):
    return db.query(models.CustomRule).filter(models.CustomRule.id == rule_id).first()

def get_rules(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.CustomRule).offset(skip).limit(limit).all()

def create_rule(db: Session, rule: schemas.CustomRuleCreate):
    db_rule = models.CustomRule(name=rule.name, pattern=rule.pattern, description=rule.description, version=1)
    db.add(db_rule)
    db.commit()
    db.refresh(db_rule)
    return db_rule

def update_rule(db: Session, rule_id: int, rule: schemas.CustomRuleUpdate):
    db_rule = get_rule(db, rule_id)
    if db_rule:
        db_rule.name = rule.name
        db_rule.pattern = rule.pattern
        db_rule.description = rule.description
        db_rule.version += 1
        db.commit()
        db.refresh(db_rule)
    return db_rule

def delete_rule(db: Session, rule_id: int):
    db_rule = get_rule(db, rule_id)
    if db_rule:
        db.delete(db_rule)
        db.commit()
    return db_rule
