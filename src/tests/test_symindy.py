from pathlib import Path
from unittest import TestCase

import deap
import matplotlib
import numpy as np
import pysindy
from sklearn.metrics import r2_score

from symindy.symindy import SymINDy


class TestSymINDy(TestCase):

    def setUp(self):
        self.symindy = SymINDy()
        inp_dir = Path(__file__).parents[1].joinpath("test_resources")
        with open(inp_dir.joinpath('x_train.test_txt'), 'r') as fl:
            self.x_tr = np.loadtxt(fl)
        with open(inp_dir.joinpath('x_test.test_txt'), 'r') as fl:
            self.x_te = np.loadtxt(fl)
        with open(inp_dir.joinpath('x_dot_train.test_txt'), 'r') as fl:
            self.xdot_tr = np.loadtxt(fl)
        with open(inp_dir.joinpath('x_dot_test.test_txt'), 'r') as fl:
            self.xdot_te = np.loadtxt(fl)
        with open(inp_dir.joinpath('time_train.test_txt'), 'r') as fl:
            self.time_tr = np.loadtxt(fl)
        with open(inp_dir.joinpath('time_test.test_txt'), 'r') as fl:
            self.time_te = np.loadtxt(fl)
        # fit SymINDy on test data
        self.symindy.fit(self.x_tr, self.xdot_tr, self.time_tr)

    def test_configure_DEAP(self):
        toolbox, creator, pset, history = self.symindy.configure_DEAP()
        self.assertTrue(isinstance(toolbox, deap.base.Toolbox))
        self.assertTrue(isinstance(pset, deap.gp.PrimitiveSetTyped))
        self.assertTrue(isinstance(history, deap.tools.History))

    def test_fit(self):
        # retrieve final SINDy model and assert it is fitted SINDy model
        final_model = self.symindy.final_model
        self.assertTrue(isinstance(final_model, pysindy.pysindy.SINDy))
        self.assertTrue(hasattr(final_model, "n_output_features_"))

    def test_predict(self):
        x_te_pred, xdot_te_pred = self.symindy.predict(self.x_te[0], self.time_te)
        self.assertEqual(self.x_te.shape, x_te_pred.shape)
        self.assertEqual(self.xdot_te.shape, xdot_te_pred.shape)
        # assert that real and predicted test data are related via R2 coefficient
        self.assertTrue(r2_score(self.x_te, x_te_pred) >0.7)
        self.assertTrue(r2_score(self.xdot_te, xdot_te_pred) >0.7)

    def test_score(self):
        x_te_pred, xdot_te_pred = self.symindy.predict(self.x_te[0], self.time_te)
        round = lambda x: np.round(x, 5)
        score_symindy_x, score_symindy_xdot = map(round, self.symindy.score(self.x_te, x_te_pred,
            self.xdot_te, xdot_te_pred))
        score_x = round(r2_score(self.x_te, x_te_pred))
        score_xdot = round(r2_score(self.xdot_te, xdot_te_pred))
        self.assertEqual(score_symindy_x, score_x)
        self.assertEqual(score_symindy_xdot, score_xdot)

    def test_plot_trees(self):
        fig, ax = self.symindy.plot_trees()
        # if pygraphviz can be successfully imported
        try:
            import pygraphviz
            pygraphviz_works = True
        except:
            pygraphviz_works = False
        if pygraphviz_works:
            self.assertTrue(isinstance(fig, matplotlib.figure.Figure))
        else:
            self.assertEqual(fig, None)
