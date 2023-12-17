# voxel-rs

## Internal Module Dependencies

```mermaid
flowchart LR
    main(main.rs) --> game
    game --> core
    game --> graphics
    game --> systems
    game --> world
    graphics --> core
    graphics --> world
    systems --> graphics
    systems --> world
```
