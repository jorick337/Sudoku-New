using System.Collections.Generic;
using Game.Classes;
using Game.Managers;
using Help.Classes;
using Help.UI;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;
using static Game.Managers.Help.GridAdd;

namespace Game.Panels
{
    public class SpesialButtonsPanel : MonoBehaviour
    {
        #region CORE

        [Header("Core")]
        [SerializeField] private Button comeBackButton;
        [SerializeField] private Button clearCellButton;
        [SerializeField] private Button hintButton;
        [SerializeField] private Button notepadButton;
        [SerializeField] private Image notepadImage;
        [SerializeField] private Button quickNotesButton;

        [Header("Focus Cell Input")]
        [SerializeField] private Button[] cellValueButtons = new Button[9]; // Кнопок всего девять

        private UnityAction[] buttonActions;

        [Header("Managers")]
        [SerializeField] private GridManager gridManager;
        [SerializeField] private HintManager hintManager;

        #endregion

        #region MONO

        private void Awake()
        {
            InitializeValues();
        }

        private void OnEnable()
        {
            RegisterEvents(true);
        }

        private void OnDisable()
        {
            RegisterEvents(false);
        }

        #endregion

        #region INITIALIZATION

        private void InitializeValues()
        {
            buttonActions = new UnityAction[9];
        }

        private void RegisterEvents(bool register)
        {
            if (register)
            {
                RegisterCellValueButtonsListeners(true);

                comeBackButton.onClick.AddListener(UndoLastMove);
                clearCellButton.onClick.AddListener(ClearFocusedCell);
                hintButton.onClick.AddListener(GenerateHint);
                notepadButton.onClick.AddListener(ToggleNotepadMode);
                quickNotesButton.onClick.AddListener(PopulateQuickNotes);
            }
            else
            {
                RegisterCellValueButtonsListeners(false);

                comeBackButton.onClick.RemoveListener(UndoLastMove);
                clearCellButton.onClick.RemoveListener(ClearFocusedCell);
                hintButton.onClick.RemoveListener(GenerateHint);
                notepadButton.onClick.RemoveListener(ToggleNotepadMode);
                quickNotesButton.onClick.RemoveListener(PopulateQuickNotes);
            }
        }

        #endregion

        #region COME BACK

        private void UndoLastMove()
        {
            Stack<Movement> movesHistory = gridManager.GridBlocks.MovesHistory;
            if (movesHistory.Count == 0)
                return;

            Movement lastMovement = movesHistory.Pop();
            Cell lastCell = gridManager.GridBlocks.Blocks[lastMovement.Block].CellManagers[lastMovement.Number].Cell;

            lastCell.SetValue(0);
            RestoreCellToGrid(gridManager, lastCell);

            if (movesHistory.Count > 0)
            {
                Movement previousMovement = movesHistory.Peek();
                Cell previousCell = gridManager.GridBlocks.Blocks[previousMovement.Block].CellManagers[previousMovement.Number].Cell;

                previousCell.SetValue(previousMovement.Value);
                RestoreCellToGrid(gridManager, previousCell);
            }
        }

        private void RestoreCellToGrid(GridManager gridManager, Cell cell)
        {
            CellManager cellManager = gridManager.GridBlocks.Blocks[cell.Block].CellManagers[cell.Number];

            cellManager.CellAdd.AddValueWithoutMoveAndChecks(cellManager, cell.Value);
            gridManager.GridAdd.AddScoreByScoreType(gridManager, ScoreType.RevertMove);
        }

        #endregion

        #region CLEAR CELL

        private void ClearFocusedCell() => gridManager.GridBlocks.FocusedCellManager.SetTextDirectly("");

        #endregion

        #region USE HINT

        private void GenerateHint()
        {
            Record record = gridManager.Sudoku.Record;
            record.SetNumberOfHints(record.NumberOfHints + 1);

            hintManager.GenerateHint();
            gridManager.GridAdd.AddScoreByScoreType(gridManager, ScoreType.HintTaken);
        }

        #endregion

        #region NOTEPAD MODE

        private void ToggleNotepadMode()
        {
            bool isNotepadMode = !gridManager.GridBlocks.IsNotepadeMode;
            float transparencyImage = (float)(isNotepadMode ? 1.0 : 0.6);

            gridManager.GridBlocks.SetIsNotepadMode(isNotepadMode);
            notepadImage.SetTransparency(transparencyImage);
        }

        #endregion

        #region QUICK NOTES

        private void PopulateQuickNotes()
        {
            gridManager.GridBlocks.SetIsNotepadMode(true);

            foreach (var cellManager in gridManager.GridBlocks.AllCellManagers)
            {
                // Если ячейка не решенная и в ней нет значений
                if (!cellManager.InputField.readOnly && cellManager.Cell.Value == 0) 
                {
                    cellManager.CellUI.SwitchToNotepadMode(cellManager);

                    Cell cell = cellManager.Cell;
                    for (int i = 1; i <= 9; i++) // Перебор подходящих значений
                    {
                        if (gridManager.Sudoku.RealGrid.IsSafe(cell.Block, cell.Number, i))
                        {
                            cellManager.SetText(i.ToString());
                        }
                    }
                }
            }

            gridManager.GridBlocks.SetIsNotepadMode(false);
        }

        #endregion

        #region FOCUS CELL BUTTONS

        private void RegisterCellValueButtonsListeners(bool canRegister)
        {
            for (int i = 0; i < cellValueButtons.Length; i++)
            {
                if (canRegister)
                {
                    int value = i + 1;
                    buttonActions[i] = () => gridManager.GridBlocks.SetValueFocusedCell(value.ToString());
                    cellValueButtons[i].onClick.AddListener(buttonActions[i]);
                }
                else
                    cellValueButtons[i].onClick.RemoveListener(buttonActions[i]);
            }
        }

        #endregion
    }
}