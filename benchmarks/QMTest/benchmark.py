from qm.fields import *
from qm.executable import *
from qm.test.resource import Resource
from qm.test.classes.command import ExecTest
from docutils.core import *
from docutils.nodes import *

class CaseExtractor(SparseNodeVisitor):

    def __init__(self, document):

        SparseNodeVisitor.__init__(self, document)
        self.cases = None
        
    def visit_paragraph(self, node):

        if node.astext() == 'Cases:':
            self.cases = {}


    def visit_field_list(self, node):
        """Cases are expressed via field-lists."""

        if self.cases or self.cases is None:
            return

        for field in node:
            name, body = field.children
            self.cases[name.astext()] = body.astext()


class BenchmarkExecutable(Resource):
    """A BenchmarkExecutable compiles a benchmark so it can be ran
    by the Benchmark."""

    executable = TextField(computed="true")
    src = TextField(computed="true")

    def GetCases(self):
        """Report benchmark cases for this benchmark resource.
        Cases may be embedded in a source file preamble (a comment block
        prefixed with '///', and using ReST markup).
        By default, a 'default' case is reported for option '-1'."""

        src = open(self.src).readlines()
        # Find the first block of '///' comment.
        while src and not src[0].startswith('///'):
            del src[0]
        if not src:
            return {'default': '-1'}
        i = 0
        while src and src[i].startswith('///'):
            i += 1
        src = src[:i]
        block = ''.join([line[3:] for line in src])
        doctree = publish_doctree(block)
        extractor = CaseExtractor(doctree)
        doctree.walk(extractor)
        return extractor.cases or {'default': '-1'}


    def SetUp(self, context, result):
        
        cmd = ['make', '-C', '..', os.path.join('benchmarks', self.executable)]
        make = RedirectedExecutable()
        # Hack: The benchmark database lives in $(builddir)/benchmarks/,
        #       but the build system in $(builddir).
        status = make.Run(cmd)
        result.CheckExitStatus('%s.'%self.GetId(), ' '.join(cmd), status)
        context['CompiledResource.executable'] = self.executable

    def CleanUp(self, result): pass


class Benchmark(ExecTest):
    
    case = TextField()

    def __init__(self, *args, **kwds):
        ExecTest.__init__(self, *args, **kwds)
        # map 'case' to 'arguments'
        if self.case:
            self.arguments = self.arguments + [self.case]

    def ValidateOutput(self, stdout, stderr, result):
        # Anything written to stderr counts as an error.
        # stdout captures performance data, and needs to be postprocessed.

        causes = []
        if self.stderr:
            causes.append("standard error")

        return causes
