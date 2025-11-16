package main;

import com.eudycontreras.othello.capsules.AgentMove;
import com.eudycontreras.othello.capsules.Index;

/**
 * Custom AgentMove implementation for the Alpha-Beta Pruning Minimax agent.
 * This class stores the move information and its evaluation value.
 */
public class MyAgentMove extends AgentMove {
    
    private Index moveIndex;
    private double value;
    
    /**
     * Default constructor
     */
    public MyAgentMove() {
        this.moveIndex = null;
        this.value = 0;
    }
    
    /**
     * Constructor with move index
     * @param index The board index for this move
     */
    public MyAgentMove(Index index) {
        this.moveIndex = index;
        this.value = 0;
    }
    
    /**
     * Constructor with move index and value
     * @param index The board index for this move
     * @param value The evaluation value of this move
     */
    public MyAgentMove(Index index, double value) {
        this.moveIndex = index;
        this.value = value;
    }
    
    /**
     * Set the move index
     * @param index The board index for this move
     */
    public void setMove(Index index) {
        this.moveIndex = index;
    }
    
    /**
     * Set the evaluation value
     * @param value The evaluation value
     */
    public void setValue(double value) {
        this.value = value;
    }
    
    /**
     * Get the evaluation value
     * @return The evaluation value
     */
    public double getValue() {
        return this.value;
    }
    
    @Override
    public Index getMoveIndex() {
        return moveIndex;
    }
    
    @Override
    public boolean isValid() {
        return moveIndex != null;
    }
    
    @Override
    public int compareTo(AgentMove other) {
        if (other instanceof MyAgentMove) {
            return Double.compare(this.value, ((MyAgentMove) other).value);
        }
        return 0;
    }
    
    @Override
    public String toString() {
        if (moveIndex != null) {
            return "Move[" + moveIndex.getRow() + "," + moveIndex.getCol() + "] = " + value;
        }
        return "Invalid Move";
    }
}
