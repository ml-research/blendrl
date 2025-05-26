import os

from nsfr.facts_converter import FactsConverter
from neumann.message_passing import MessagePassingModule
from neumann.reasoning_graph import ReasoningGraphModule
from nsfr.utils.logic import get_lang, get_blender_lang, build_infer_module
from neumann.neumann import NEUMANN
from nsfr.valuation import ValuationModule
from neumann.soft_logic import SoftLogic


def get_neumann_model(env_name: str, rules: str, device: str, train=True, explain=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = f"in/envs/{env_name}/logic/"

    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, rules)

    val_fn_path = f"in/envs/{env_name}/valuation.py"
    val_module = ValuationModule(val_fn_path, lang, device)

    # FC = FactsConverter(lang=lang, valuation_module=val_module, atoms=atoms, bk=bk, device=device)
    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(prednames)
    # m = 5
    # IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)
    # Neuro-Symbolic Forward Reasoner
    soft_logic = SoftLogic()
    MPM = MessagePassingModule(soft_logic=soft_logic, device=device, T=2)
    RGM = ReasoningGraphModule(clauses=clauses, facts=atoms, terms=lang.consts, lang=lang, max_term_depth=1, device=device)  
    neumann = NEUMANN(facts_converter=FC, message_passing_module=MPM, reasoning_graph_module=RGM, program_size=m, atoms=atoms, bk=bk, clauses=clauses, device=device, train=train, explain=explain)
    return neumann


def get_blender_neumann_model(env_name: str, rules: str, device: str, train=False, mode='normal'):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = f"in/envs/{env_name}/logic/"

    lang, clauses, bk, atoms = get_blender_lang(lark_path, lang_base_path, rules)

    val_fn_path = f"in/envs/{env_name}/valuation.py"
    val_module = ValuationModule(val_fn_path, lang, device)

    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    # if train:
    #     m = len(prednames)
    # else:
    #     m = len(clauses)
    m = len(clauses)
    # m = 5
    # Neuro-Symbolic Forward Reasoner
    soft_logic = SoftLogic()
    MPM = MessagePassingModule(soft_logic=soft_logic, device=device, T=2)
    RGM = ReasoningGraphModule(clauses=clauses, facts=atoms, terms=lang.consts, lang=lang, max_term_depth=1, device=device)  
    neumann = NEUMANN(facts_converter=FC, message_passing_module=MPM, reasoning_graph_module=RGM, program_size=m, atoms=atoms, bk=bk, clauses=clauses, train=train)
    return neumann