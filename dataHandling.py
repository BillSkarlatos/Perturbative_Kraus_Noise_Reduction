import numpy as np

def format_matrix_to_latex(matrix, gate_name):
    """
    Format a matrix into a LaTeX bmatrix representation.
    """
    latex_matrix = "\\begin{bmatrix}\n"
    for row in matrix:
        latex_matrix += " & ".join(f"{np.real(elem):.5f}" for elem in row) + " \\\\\n"
    latex_matrix += "\\end{bmatrix}"
    
    return f"\\textbf{{Gate: {gate_name}}}\n\n{latex_matrix}\n"