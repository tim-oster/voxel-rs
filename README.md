# voxel-rs

## Internal Module Dependencies

```mermaid
flowchart LR
    main(main.rs) --> gamelogic
    gamelogic --> core
    gamelogic --> graphics
    gamelogic --> systems
    gamelogic --> world
    graphics --> core
    graphics --> world
    systems --> graphics
    systems --> world
```
