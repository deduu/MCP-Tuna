# AGENTS.md

Rules for the React dashboard in `frontend/`.

- All backend calls go through `src/api/` clients and hooks. Do not fetch raw pipeline endpoints directly from components.
- Keep feature pages thin. Move branching logic into hooks, `src/lib/`, or small domain components.
- Prefer capability-driven UI. Read advertised capabilities before exposing controls or assumptions.
- UI primitives belong in `src/components/ui/`. Domain-specific behavior belongs in the matching feature folder.
- Add new routes in `src/App.tsx` and update navigation or quick actions when the feature should be discoverable.
- Use Tailwind classes and existing UI primitives instead of inline styles or one-off patterns.
- Keep user feedback in the established toast and status components, not `alert()` or ad hoc console output.
- Verification: run `npm --prefix frontend run build`.
