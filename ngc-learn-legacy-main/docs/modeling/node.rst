Node
====

A Node represents one of the fundamental building blocks of an NGC system. These
particular objects are meant to perform, per simulated time step, a calculation of
output activity values given an internal arrangement of compartments (or sources
where signals from other Node(s) are to be deposited).

.. _node-model:

Node Model
--------------
The ``Node`` class serves as a root class for the node building block objects of an
NGC system/graph. This is a core modeling component of general NGC computational
systems. Node sub-classes within ngc-learn inherit from this base class.

.. autoclass:: ngclearn.engine.nodes.node.Node
  :noindex:

  .. automethod:: wire_to
    :noindex:
  .. automethod:: inject
    :noindex:
  .. automethod:: clamp
    :noindex:
  .. automethod:: step
    :noindex:
  .. automethod:: calc_update
    :noindex:
  .. automethod:: clear
    :noindex:
  .. automethod:: extract
    :noindex:
  .. automethod:: extract_params
    :noindex:
  .. automethod:: deep_store_state
    :noindex:


.. _snode-model:

SNode Model
-----------
The ``SNode`` class extends from the base ``Node`` class, and represents
a (rate-coded) state node that follows a certain set of settling dynamics.
In conjunction with the corresponding ``ENode`` and ``FNode`` classes,
this serves as the core modeling component of a higher-level ``NGCGraph`` class
used in simulation.

.. autoclass:: ngclearn.engine.nodes.snode.SNode
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: clear
    :noindex:


.. _enode-model:

ENode Model
-----------
The ``ENode`` class extends from the base ``Node`` class, and represents
a (rate-coded) error node simplified to its fixed-point form.
In conjunction with the corresponding ``SNode`` and ``FNode`` classes,
this serves as the core modeling component of a higher-level ``NGCGraph`` class
used in simulation.

.. autoclass:: ngclearn.engine.nodes.enode.ENode
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: calc_update
    :noindex:
  .. automethod:: compute_precision
    :noindex:
  .. automethod:: clear
    :noindex:


.. _fnode-model:

FNode Model
-----------
The ``FNode`` class extends from the base ``Node`` class, and represents
a stateless node that simply aggregates (via summation) its received inputs.
In conjunction with the corresponding ``SNode`` and ``ENode`` classes,
this serves as the core modeling component of a higher-level ``NGCGraph`` class
used in simulation.

.. autoclass:: ngclearn.engine.nodes.fnode.FNode
  :noindex:

  .. automethod:: step
    :noindex:

SpNode_Enc Model
----------------
The ``SpNode_Enc`` class extends from the base ``Node`` class, and represents
a spiking state node that follows settling dynamics according a rate-code
probabilistic spike process (a Poisson process).
In conjunction with the corresponding the other node classes,
this serves as the core modeling component of a higher-level ``NGCGraph`` class
used in simulation.

.. autoclass:: ngclearn.engine.nodes.spiking.spnode_enc.SpNode_Enc
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: clear
    :noindex:

SpNode_LIF Model
----------------
The ``SpNode_LIF`` class extends from the base ``Node`` class, and represents
a spiking state node that follows settling dynamics according the mathematical
model of the leaky integrate and fire (LIF) neuron.
In conjunction with the corresponding the other node classes,
this serves as the core modeling component of a higher-level ``NGCGraph`` class
used in simulation.

.. autoclass:: ngclearn.engine.nodes.spiking.spnode_lif.SpNode_LIF
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: clear
    :noindex:

SpNode_ELIF Model
-----------------
The ``SpNode_ELIF`` class extends from the ``SpNode_LIF`` class, and represents
a spiking state node that follows settling dynamics according the mathematical
model of the exponential leaky integrate and fire (ELIF) neuron.
In conjunction with the corresponding the other node classes,
this serves as the core modeling component of a higher-level ``NGCGraph`` class
used in simulation.

.. autoclass:: ngclearn.engine.nodes.spiking.spnode_elif.SpNode_ELIF
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: clear
    :noindex:
